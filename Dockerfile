FROM node:20-bookworm

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# 1. Create the virtual environment in /opt/venv (safe from bind mounts)
RUN python3 -m venv /opt/venv

# 2. Add the safe venv to the system PATH
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace

RUN npm install -g @google/gemini-cli

# 3. Copy requirements and install (it will use /opt/venv automatically)
COPY requirements.txt .
RUN pip install -r requirements.txt && rm requirements.txt

ENV TERM=xterm-256color
ENV GIT_AUTHOR_NAME=metal-thread
ENV GIT_COMMITTER_NAME=metal-thread
ENV GIT_AUTHOR_EMAIL=aragonn@berkeley.edu
ENV GIT_COMMITTER_EMAIL=aragonn@berkeley.edu
