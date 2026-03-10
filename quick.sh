# python tools/synthesize_dialogue.py \
#        --model-dir pretrained_models/Fun-CosyVoice3-0.5B-2512 \
#        --dialogue-dir syn_data/dialogues \
#        --speaker-dir syn_data/ref_audios \
#        --output-dir syn_data/outputs \
#        --concurrency 2

# for i in {0..1}; do
#   python tools/synthesize_dialogue.py \
#     --model-dir pretrained_models/Fun-CosyVoice3-0.5B-2512 \
#     --dialogue-dir syn_data/WardRound_dialogues \
#     --speaker-dir syn_data/ref_audios \
#     --output-dir syn_data/WardRound_outputs \
#     --concurrency 2 \
#     --worker-id $i \
#     --num-workers 2 \
#     --speed 1
# done

#python tools/compose_wav.py --input-dir syn_data/outputs/test_001/ --output-path syn_data/outputs/test_001/compose.wav

#python tools/batch_compose_dialogues.py \
#     --input-dir syn_data/WardRound_outputs \
#     --output-dir syn_data/WardRound_outputs/composed_wavs \
#     --silence-duration 0.6 \
#     --pattern "ward_round_dialogue_*" 


#python tools/extract_dialogue_text.py \
#    --input-dir syn_data/Surgery_dialogues \
#    --output-dir syn_data/Surgery_dialogues/composed_texts \
#    --pattern "surgery_dialogue_*" \
#    --quiet

python tools/synthesize_dialogue_en.py \
    --model-dir pretrained_models/Fun-CosyVoice3-0.5B-2512 \
    --dialogue-dir syn_data_en/Surgery_dialogues \
    --speaker-dir syn_data_en/ref_audios \
    --output-dir syn_data_en/Surgery_outputs \
    --mode cross_lingual \
    --fp16
