#!/bin/sh

if ! command -v rsync &> /dev/null; then
  echo "rsync required, but not installed!"
  exit 1
else
  rsync -avh nomad-external-eln-integrations/ .
  rm -rfv nomad-external-eln-integrations
fi
