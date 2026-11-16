#!/usr/bin/env bash
# Step 10, part 1: a local Kubernetes cluster (kind) with our stack on it.  ~5-8 min.
#
#   bash scripts/setup_kind.sh          # install tools, create cluster, load images, deploy
#   kind delete cluster --name llm      # remove everything afterwards
set -euo pipefail
CLUSTER=llm
KIND_VERSION=v0.33.0
LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server
REDIS_IMAGE=redis:7-alpine
GW_IMAGE=llm-gateway:step10
mkdir -p "$HOME/.local/bin"; export PATH="$HOME/.local/bin:$PATH"

echo "########## 1. tools ##########"
if ! command -v kind > /dev/null; then
  curl -sSLo "$HOME/.local/bin/kind" "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-linux-amd64"
  chmod +x "$HOME/.local/bin/kind"
fi
if ! command -v kubectl > /dev/null; then
  curl -sSLo "$HOME/.local/bin/kubectl" "https://dl.k8s.io/release/$(curl -sSL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
  chmod +x "$HOME/.local/bin/kubectl"
fi
kind version; kubectl version --client | head -1

echo; echo "########## 2. free the machine (Step 9 stack uses the same CPUs) ##########"
docker compose down > /dev/null 2>&1 || true
docker rm -f llamacpp redis > /dev/null 2>&1 || true
pkill -f "[u]vicorn app:app" 2>/dev/null || true
[[ -f models/qwen2.5-0.5b-instruct-fp16.gguf ]] || { echo "missing model: python scripts/download_gguf.py fp16"; exit 1; }

echo; echo "########## 3. images: build the gateway, make sure the others are local ##########"
# --provenance=false: no extra "attestation" entries in the image index (they confuse the import below)
docker build -q --provenance=false -t "$GW_IMAGE" serving/gateway > /dev/null 2>&1 \
  || docker build -q -t "$GW_IMAGE" serving/gateway > /dev/null
echo "built $GW_IMAGE"
docker image inspect "$LLAMA_IMAGE" > /dev/null 2>&1 || docker pull -q "$LLAMA_IMAGE"
docker image inspect "$REDIS_IMAGE" > /dev/null 2>&1 || docker pull -q "$REDIS_IMAGE"

echo; echo "########## 4. cluster ##########"
if ! kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  kind create cluster --name "$CLUSTER" --config deploy/k8s/kind-config.yaml --wait 120s
fi
kubectl cluster-info --context "kind-$CLUSTER" | head -1
SERVER_MINOR=$(kubectl version -o json | python -c "import json,sys; print(int(json.load(sys.stdin)['serverVersion']['minor'].rstrip('+')))")
(( SERVER_MINOR >= 30 )) || { echo "Kubernetes 1.$SERVER_MINOR is too old (need 1.30+ for preStop sleep + ContainerResource HPA)"; exit 1; }
echo "Kubernetes 1.$SERVER_MINOR"

echo; echo "########## 5. load images into the node (it can't see the codespace's Docker images) ##########"
# Not `kind load docker-image`: with Docker's newer image store it imports ALL platforms and fails
# ("content digest ... not found") because only linux/amd64 is on disk. Instead: export just our
# platform and import it into the node's containerd ourselves.
NODE="${CLUSTER}-control-plane"
node_has() { docker exec "$NODE" crictl inspecti "$1" > /dev/null 2>&1; }
import_into_node() {   # $1 = image; $2 = 1 to allow falling back to a registry pull
  local img="$1"
  if node_has "$img"; then echo "  $img: already on the node"; return 0; fi
  for save in "docker save --platform linux/amd64 $img" "docker save $img"; do
    if $save 2>/dev/null | docker exec -i "$NODE" ctr --namespace=k8s.io images import \
         --digests --snapshotter=overlayfs - > /dev/null 2>&1 && node_has "$img"; then
      echo "  $img: imported ($save)"; return 0
    fi
  done
  if [[ "${2:-0}" == "1" ]]; then
    echo "  $img: import failed -> the node pulls it from the registry instead"
    docker exec "$NODE" crictl pull "$img" > /dev/null && node_has "$img" && { echo "  $img: pulled"; return 0; }
  fi
  echo "  ERROR: could not get $img onto the node"; return 1
}
import_into_node "$GW_IMAGE" 0          # local-only image: must be imported
import_into_node "$LLAMA_IMAGE" 1
import_into_node "$REDIS_IMAGE" 1

echo; echo "########## 6. metrics-server (the HPA's source of CPU numbers) ##########"
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml > /dev/null
# kind's kubelets use self-signed certificates -> tell metrics-server not to verify them (local only!)
if ! kubectl -n kube-system get deploy metrics-server -o jsonpath='{.spec.template.spec.containers[0].args}' | grep -q insecure; then
  kubectl -n kube-system patch deployment metrics-server --type=json \
    -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]' > /dev/null
fi
kubectl -n kube-system rollout status deployment/metrics-server --timeout=180s

echo; echo "########## 7. deploy: namespace, Redis, llm (2 replicas, no autoscaler yet) ##########"
kubectl apply -f deploy/k8s/namespace.yaml -f deploy/k8s/redis.yaml -f deploy/k8s/llm.yaml
kubectl -n llm rollout status deployment/redis --timeout=120s
kubectl -n llm rollout status deployment/llm --timeout=300s
kubectl -n llm get pods -o wide

echo; echo "########## 8. how much of the node's CPU is reserved ##########"
kubectl describe node "${CLUSTER}-control-plane" | sed -n '/Allocated resources/,/Events/p' | head -8
echo; echo "cluster ready. Next: bash scripts/run_k8s_checks.sh"