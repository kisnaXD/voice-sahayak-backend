import os
from google.cloud import texttospeech_v1 as texttospeech

# --------- CONFIG ---------
LANG_CODES = {
    "en": "en-IN",
    "hi": "hi-IN",
    "bn": "bn-IN",
    "te": "te-IN",
    "mr": "mr-IN",
    "ta": "ta-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
}

OUTPUT_DIR = "prompt_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------- STATIC PROMPTS (TEXT) ---------
# You can adjust wording if needed.  
# These texts will be synthesized in all 9 languages.

PROMPTS = {
    "no_input": "We did not receive any input.",
    "invalid_selection": "Invalid selection. Please try again.",
    "did_not_receive_name_try_one_more_time": "We did not receive your name. Let us try one more time.",
    "still_no_name_continue_without": "We still did not receive your name. We will continue without it.",
    "could_not_hear_name_try_again": "We could not hear your name properly. Let us try once again.",
    "did_not_receive_aadhaar_try": "We did not receive your Aadhaar number. Let us try again.",
    "still_no_aadhaar_goodbye": "We still did not receive your Aadhaar number. Goodbye.",
    "aadhaar_too_short": "The number entered seems too short. Please try again.",
    "sorry_not_found": "We could not find any notice for the provided Aadhaar number.",
    "ask_if_want_to_explain_source": "If you want, you can explain where the notice is from and we will try again.",
    "no_input_detected_goodbye": "No input detected. Goodbye.",
    "notice_not_readable": "We could not read the notice properly.",
    "please_wait_summary": "Please wait while we prepare your summary.",
    "error_generating": "We faced an error while generating your summary.",
    "thanks_we_will_try_search_call_back": "Thanks. We will try searching with that information and call you back."
}

# --------- TTS CLIENT ---------
client = texttospeech.TextToSpeechClient()

def generate_mp3(text, lang_code, output_path):
    synthesis_input = texttospeech.SynthesisInput(text=text)
    if(lang_code == 'hi-IN'):
        voice = texttospeech.VoiceSelectionParams(
            language_code=lang_code,
            name="hi-IN-Wavenet-A"
        )
    else:
        voice = texttospeech.VoiceSelectionParams(
            language_code=lang_code
        )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    with open(output_path, "wb") as f:
        f.write(response.audio_content)
    print(f"Generated: {output_path}")

def main():
    for lang_key, gcode in LANG_CODES.items():
        for message_name, text in PROMPTS.items():
            filename = f"{lang_key}_{message_name}.mp3"
            full_path = os.path.join(OUTPUT_DIR, filename)
            generate_mp3(text, gcode, full_path)

    print("\nAll prompt MP3s generated successfully!")

if __name__ == "__main__":
    main()
