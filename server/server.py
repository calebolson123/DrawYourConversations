from flask import Flask, request
import os 
import io
import sys
import subprocess
import glob
import json
import asyncio
import random
from base64 import encodebytes
from transformers import pipeline
import spacy
import whisper

from gphotospy import authorize
from gphotospy.media import Media
from gphotospy.album import Album

import modifiers
import PIL
from PIL import Image, ImageDraw, ImageFont, ImageOps

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

lock = False
app = Flask(__name__)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
spacy_nlp = spacy.load("en_core_web_lg")
spacy_nlp.add_pipe('sentencizer')

album_title = 'living_room' # human readable name of your album in google photos
sentiment_pipeline = pipeline(model='cardiffnlp/twitter-roberta-base-sentiment') # can switch this up, but this one worked well

@app.route("/")
def hello_world():
    return "alive"


# If you wish to auto-backup images created as seen in my YouTube video,
# you'll need to set up auth against google. Start here: https://developers.google.com/photos/library/guides/get-started

# To use Chromecast's personal album feature (to have live slideshow of images generated, on your Chromecast), it requires a min of 5 photos.
# This is undocumented from what I can tell. This func will backup and maintain the min required (5) for the feature to work. It removes the oldest
# photo whenever a new one is generated, maintaining a live album of the 5 most recently generated images
async def backup_image(path_to_img):
    print("Backing up image.")
    CLIENT_SECRET_FILE = "credentials.json" # you'll need to create this file & store in this dir
    service = authorize.init(CLIENT_SECRET_FILE)

    # Init the album and media manager
    album_manager = Album(service)
    media_manager = Media(service)

    album_iterator = album_manager.list()

    album_id = None
    keep_looking = True
    while(keep_looking):
        try:
            # Print only album's title (if present, otherwise None)
            item = next(album_iterator)
            print(item.get("title"))
            if item.get("title") == album_title:
                print("found album!")
                album_id = item.get("id")
        except (StopIteration, TypeError) as e:
            # Handle exception if there are no albums left
            keep_looking = False
            print("No (more) albums.")
            break

    media_manager.stage_media(path_to_img)
    try:
        media_manager.batchCreate(album_id)
    except Exception as e:
        print("CAUGHT: ", e)
    
    media_iterator = media_manager.list()
    search_iterator = media_manager.search_album(album_id)
    should_delete = True
    for _ in range(5):
        try:
            # Print only media's filename (if present, otherwise None)
            print(next(search_iterator).get("filename"))
            pass
        except (StopIteration, TypeError) as e:
            should_delete = False
            # Handle exception if there are no media left
            print("No (more) media in album.")
            break

    if should_delete:
        try:
            print(next(media_iterator).get("filename"))
            item = next(search_iterator)
            to_delete = [item.get("id")]
            print("to_delete: ", item.get("filename"), " : ", to_delete)
            album_manager.batchRemoveMediaItems(album_id, to_delete)
        except (StopIteration, TypeError) as e:
            # Handle exception if there are no media left
            print("No (more) media.")
            pass

    print("Backup complete.")


def get_response_image(image_path):
    pil_img = PIL.Image.open(image_path, mode='r')
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img


@app.route('/draw', methods=["POST"])
async def draw():
    print("draw")
    global lock
    if lock == True:
        return "Processing...", 503
    lock = True

    conversation = request.json['conversation']
    print("Raw conversation received: ", conversation)

    if len(conversation.split()) > 50: # summarize
        print("Summarizing.")
        summarization = summarizer(conversation, max_length=30, min_length=5, do_sample=False)[0]
        print("\nsummarization obj: ", summarization)
        summarization_text = summarization['summary_text']
        print("\nsummarization text: ", summarization_text)
    else: # don't summarize
        print("Not Summarizing.")
        summarization_text = conversation

    # 0 -> Negative -> "adj-horror"
    # 1 -> Neutral -> "adj-general"
    # 2 -> Positive -> "adj-beauty"
    summarization_text = summarization_text.replace('"', "") 

    print("sentiment: ", sentiment_pipeline(summarization_text))

    sentiment_class = sentiment_pipeline(summarization_text)[0]['label']

    prompt = ''
    doc = spacy_nlp(summarization_text)

    # split into sentences and add comma at end of sentence
    included_tags = {"NOUN", "VERB", "ADV" "ADJ", "PROPN"}
    excluded = {'"', "'", "'s", ",", "?"}
    swap_for_comma = {}

    # add random modifiers to tokens based on tag type
    for sent_i, sent in enumerate(doc.sents):
        modifier_used = False # only allow one injected modifier per "sentence"
        print("sentence: ", sent)
        add_comma = False

        for token in sent: # if this sentence already contains an adj, don't inject additional
            if token.pos_ == 'ADJ':
                modifier_used = True
            token.text.replace('?', ',')

        for token in sent:
            print("token in sentence: ", token, ' _type: ', token.pos_, ' : ', token.text, ' : ', token.is_stop)

            if token.text in swap_for_comma and prompt[-1] != ',':
                prompt += ', '
                continue
            
            if token.pos_ in included_tags and token.text not in prompt and token.text not in excluded:
                print("include token: ", token)
                if (token.pos_ == "NOUN" or token.pos_ == "PROPN") and not modifier_used and sent_i >= 1:
                    modifier_used = True
                    print("add adjective before this word: ", token)
                    adj = random.choice(modifiers.nspterminology['adj-general'])
                    if sentiment_class == "LABEL_0":
                        adj = random.choice(modifiers.nspterminology['adj-horror'])
                    elif sentiment_class == "LABEL_2":
                        adj = random.choice(modifiers.nspterminology['adj-beauty'])

                    print("injecting adj: ", adj)
                    prompt += adj + ' '

                prompt += token.text + ' '

                add_comma = True
        if add_comma:
            prompt = prompt.strip() + ', '

    print("\n\n")

    # add rng flavor
    summarization_text = prompt
    if random.random() <= .5:
        summarization_text += ' ' + random.choice(modifiers.nspterminology['details']) + ','
    if random.random() <= .40:
        summarization_text += ' ' + random.choice(modifiers.nspterminology['site']) + ', ' + random.choice(modifiers.nspterminology['hd'])
    elif random.random() <= .40:
        summarization_text += 'by ' + random.choice(modifiers.nspterminology['artist'])
    elif random.random() <= .40:
        summarization_text += ' ' + random.choice(modifiers.nspterminology['style'])

    if summarization_text[-1] == ',':
        summarization_text = summarization_text[:-1]

    # summarization_text = conversation # uncomment this line to override above prompt engineering, and to pass input convo directly into SD
    print("\nSummarization going into model: ", summarization_text)
    arguments = " --outdir ../out --n_iter 1 --n_samples 1 --plms --skip_grid --ddim_steps 200 --scale 20 --W 384 --H 384" # you can tweak these based on your system
    subprocess.run("activate ldm & cd stable-diffusion & python scripts/txt2img.py --prompt " + "\"" + summarization_text + "\"" + arguments, shell=True) # this is very ugly, but was quickest path forward ;)

    list_of_files = glob.glob('./out/samples/*')
    latest_file = max(list_of_files, key=os.path.getctime)


    # subsequent code embeds prompt used by SD, onto the image, as seen at end of youtube video
    # Load image
    im = PIL.Image.open(latest_file)

    # Load font and work out size of summarization_text
    text = summarization_text.split()
    n = 6
    lines_to_draw = [' '.join(text[i:i+n]) for i in range(0,len(text),n)]

    # font = ImageFont.load_default()
    font = ImageFont.truetype(font=os.environ["PATH_TO_FONT_FILE"], size=14) # insert path to local font file
    tw, th = font.getsize(lines_to_draw[0])
    padding = 40

    # Extend image at bottom and get height and width of new canvas
    extended = ImageOps.expand(im, border=(0,0,0,th+2*padding), fill=(0,0,0))
    w, h = extended.size

    # Get drawing context and annotate
    draw = ImageDraw.Draw(extended)

    base_y = (h-th-padding-30)
    p = 1
    for line in lines_to_draw: # draw prompt text onto extended image
        tw, th = font.getsize(line)
        y = base_y + (15 * p)
        draw.text(((w-tw)//2, y), line, (255,255,255), font=font)
        p += 1

    extended.save(latest_file) # save to fs

    asyncio.create_task(backup_image(latest_file)) # comment this out if do not desire backing up image to google photos

    encoded_img = get_response_image(latest_file)
    json = {
        'img': encoded_img,
        'prompt': summarization_text,
    }

    print("\n\nRAW: ", conversation)
    print("\n\nPROMPT: ", summarization_text)

    lock = False
    return json


@app.route('/transcribe', methods=["POST"])
def transcribe():
    model = whisper.load_model("base.en").cuda()

    audio_path = request.json['audio_path']

    # load audio and pad/trim it to fit 30 seconds
    if audio_path == "SHARE":
        audio = whisper.load_audio(os.environ["PATH_TO_FILESHARE"])
    else:
        audio = whisper.load_audio(os.environ["BASE_PATH_TO_RECORDING"] + audio_path)

    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # decode the audio
    options = whisper.DecodingOptions(language='en')
    result = whisper.decode(model, mel, options)

    print("no_speech_prob: ", result.no_speech_prob)

    if result.no_speech_prob > .5: # likely just noise
        print('Likely just noise, dont return: ', result.text)
        return ''

    return { 'result': result.text }


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(app.run(host='0.0.0.0', port=5000, ssl_context='adhoc', debug=True, threaded=True)) # TODO: ew bad, fix
