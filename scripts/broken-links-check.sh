#!/usr/bin/env bash

# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

set -o errexit

# Function to display usage
usage() {
  echo "Usage: $0 --url <URL>"
  exit 1
}

[ "$#" -ne 2 ] && usage

while [ "$1" != "" ]; do
  case $1 in
    --url ) shift; URL=$1 ;;
    * ) usage ;;
  esac
  shift
done

echo "Checking broken links for: $URL"

# Run muffet to check for broken links
muffet \
  --buffer-size=16384 \
  --max-connections=1 \
  --color=always \
  --header="User-Agent:Mozilla/5.0(Firefox/97.0)" \
  --max-connections-per-host=1 \
  --rate-limit=1 \
  --max-response-body-size=20000000 \
  --ignore-fragments \
  --exclude="($(paste -sd '|' .muffet-excluded-links.txt))" \
  "$URL"
