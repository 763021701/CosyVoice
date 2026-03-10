for i in {501..520}; do
    num=$(printf "%04d" $i)
    if [ -d "ward_round_dialogue_${num}" ]; then
        rm -rfv "ward_round_dialogue_${num}"
    fi
done
