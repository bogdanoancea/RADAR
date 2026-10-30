#!/bin/sh
# Send biotools.json to bio.tools (creates nothing new: updates the entry radar-raman).
# Usage:  sh tools/update_biotools.sh        (asks for your bio.tools user name and password)
set -e
cd "$(dirname "$0")/.."
printf "bio.tools user name: "; read -r BT_USER
printf "bio.tools password: "; stty -echo; read -r BT_PASS; stty echo; echo
TOKEN=$(curl -s -X POST https://bio.tools/api/rest-auth/login/ \
  -H "Content-Type: application/json" \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"username": sys.argv[1], "password": sys.argv[2]}))' "$BT_USER" "$BT_PASS")" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin).get("key",""))')
[ -n "$TOKEN" ] || { echo "Login failed: check the user name and password."; exit 1; }
curl -s -X PUT https://bio.tools/api/tool/radar-raman/ \
  -H "Authorization: Token $TOKEN" -H "Content-Type: application/json" \
  --data-binary @biotools.json -w "\nHTTP status: %{http_code}\n"
