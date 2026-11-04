#!/bin/bash
# Prints "PENDING_LONG <jobid> <name> <seconds>" once per job when one of my jobs has been pending for more than an hour (fallback rule: move it to V100).
# Run locally: bash tools/pending_watch.sh
seen=""
while true; do
    out=$(ssh jz 'bash -l -s' <<'EOF'
squeue -u $USER -h -t PD -o "%i %j %V" | while read id name sub; do
    age=$(( $(date +%s) - $(date -d "$sub" +%s) ))
    echo "$id $name $age"
done
EOF
)
    while read -r id name age; do
        [ -z "$id" ] && continue
        if [ "$age" -gt 3600 ] && [[ " $seen " != *" $id "* ]]; then
            echo "PENDING_LONG $id $name ${age}s"
            seen="$seen $id"
        fi
    done <<< "$out"
    sleep 300
done
