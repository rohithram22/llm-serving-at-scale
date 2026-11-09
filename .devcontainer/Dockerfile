FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

# The base image ships a Yarn apt repo whose signing key is missing.
# That makes `apt-get update` fail, which breaks the docker-in-docker feature.
# We don't use Yarn, so remove that repo before features are installed.
RUN grep -rl "yarnpkg" /etc/apt/sources.list.d/ 2>/dev/null | xargs -r rm -f