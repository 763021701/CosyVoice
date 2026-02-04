# python tools/synthesize_dialogue.py \
#        --model-dir pretrained_models/Fun-CosyVoice3-0.5B-2512 \
#        --dialogue-dir syn_data/dialogues \
#        --speaker-dir syn_data/ref_audios \
#        --output-dir syn_data/outputs \
#        --concurrency 2

for i in {1..5}; do
  python tools/synthesize_dialogue.py \
    --model-dir pretrained_models/Fun-CosyVoice3-0.5B-2512 \
    --dialogue-dir syn_data/dialogues \
    --speaker-dir syn_data/ref_audios \
    --output-dir syn_data/outputs \
    --concurrency 2 \
    --worker-id $i \
    --num-workers 6
done

#python tools/compose_wav.py --input-dir syn_data/outputs/test_001/ --output-path syn_data/outputs/test_001/compose.wav
