#!/bin/bash
# Prints "FIRST <label> job=<id>" when the first of the given jobs of a group reaches COMPLETED, "PROBLEM <label> job=<id> state=<s>" if one ends otherwise.
# Usage: bash tools/first_done.sh "run:1927245,1927616" "jump_any:1927246,1927633"
declare -A finished
n=$#
while true; do
    for spec in "$@"; do
        label="${spec%%:*}"
        ids="${spec##*:}"
        if [ -n "${finished[$label]}" ]; then
            continue
        fi
        for j in ${ids//,/ }; do
            st=$(ssh jz "bash -l -c 'sacct -j $j -X -n -o State | head -1'" | tr -d ' ')
            if [ "$st" = "COMPLETED" ]; then
                echo "FIRST $label job=$j"
                finished[$label]=$j
                break
            elif [ -n "$st" ] && [ "$st" != "PENDING" ] && [ "$st" != "RUNNING" ] && [ "$st" != "COMPLETING" ]; then
                echo "PROBLEM $label job=$j state=$st"
            fi
        done
    done
    if [ "${#finished[@]}" -ge "$n" ]; then
        break
    fi
    sleep 30
done
echo "all groups have a finished job"
