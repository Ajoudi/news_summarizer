from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from datetime import datetime
from tqdm import tqdm
import os 

chrome_options = Options()
ARG_WINDOW_SIZE = "--window-size=1920,1080"

chrome_options.add_argument(ARG_WINDOW_SIZE)
prefs = {"profile.managed_default_content_settings.images": 2}
chrome_options.add_experimental_option("prefs", prefs)
driver = webdriver.Chrome(options=chrome_options)


driver.get('https://sso.accounts.dowjones.com/login-page?op=localop&scope=openid%20idp_id%20roles%20email%20given_name%20family_name%20djid%20djUsername%20djStatus%20trackid%20tags%20prts%20suuid%20updated_at&client_id=5hssEAdMy0mJTICnJNvC9TXEw3Va7jfO&response_type=code&redirect_uri=https%3A%2F%2Faccounts.wsj.com%2Fauth%2Fsso%2Flogin&nonce=2043a6c8-fb2b-4be8-bd16-f6a8f802c345&ui_locales=en-us-x-wsj-223-2&mars=-1&ns=prod%2Faccounts-wsj&state=7ButPrt3ezdIjpmc.2tcijS-7hsKv4FlavdVXcHEMhBVQF__SaZ6ElNjIH_M&protocol=oauth2&client=5hssEAdMy0mJTICnJNvC9TXEw3Va7jfO#!/signin-password')
time.sleep(5)
username_input = driver.find_element(By.ID, "password-login-username")
username_input.send_keys('WSJ EMAIL')

password_input = driver.find_element(By.ID, "password-login-password")
password_input.send_keys("PASSWORD")

button_list = driver.find_elements(By.TAG_NAME, 'button')
for temp_button in button_list:
    if temp_button.text == 'Sign In':
        signin_button = temp_button 
signin_button.click()


time.sleep(5)

today = datetime.today()
date_today = today.strftime('%Y-%m-%d')


newpath = './wsj_article_dfs/{}/'.format(date_today)

if not os.path.exists(newpath):
    os.makedirs(newpath)

val_date = date_today.split('-')


wsj_link_df_list = []

# fetch all links
for page_num in [1,2,3,4,5]:
    driver.get('https://www.wsj.com/news/archive/{}/{}/{}?page={}'.format(val_date[0], val_date[1], val_date[2], page_num))

    temp_df_list = []
    all_links = driver.find_elements(By.TAG_NAME, """a""")

    for i in all_links:
        try:
            val1 = i.get_attribute('href')
            val2 = i.find_element(By.TAG_NAME, 'span').text

            if 'articles' in val1: 

                temp_df_list += [{
                    'link':val1,
                    'title': val2
                }]

        except:
            pass

    temp_df = pd.DataFrame(temp_df_list) 
    if len(temp_df) > 0:
        wsj_link_df_list += temp_df_list
    else:
        break

wsj_link_df = pd.DataFrame(wsj_link_df_list) 
wsj_link_df.to_csv(newpath + 'wsj_initial_df.csv',index=False)

wsj_final_df_list = []

# fetch text from all links 
for i in range(len(wsj_link_df)):
    try:
        all_text = ''
        temp_link = wsj_link_df.iloc[i]['link']
        temp_title = wsj_link_df.iloc[i]['title']
        driver.get(temp_link)
        pgs = driver.find_elements(By.CSS_SELECTOR, "p.css-xbvutc-Paragraph.e3t0jlg0")

        for t_pg in pgs:
            all_text += t_pg.text + ' '

        if len(all_text) > 0:
            wsj_final_df_list += [{
                'title': temp_title,
                'link': temp_link, 
                'text': all_text,
                'chars': len(all_text)
            }]

        time.sleep(1)
    except:
        pass


wsj_final_df = pd.DataFrame(wsj_final_df_list)
wsj_final_df.to_csv(newpath + 'wsj_final_df.csv',index=False)

print("DONE WITH {}".format(date_today))
driver.close()


# Summarize 

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import pandas as pd
from datetime import datetime, timedelta

checkpoint = "philschmid/bart-large-cnn-samsum"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

nltk.download('punkt')

def summarize_text(text_input):
    length = 0
    chunk = ""
    chunks = []
    count = -1
    
    sentences = nltk.tokenize.sent_tokenize(text_input)
    for sentence in sentences:
        count += 1
        combined_length = len(tokenizer.tokenize(sentence)) + length # add the no. of sentence tokens to the length counter

        if combined_length  <= tokenizer.max_len_single_sentence: # if it doesn't exceed
            chunk += sentence + " " # add the sentence to the chunk
            length = combined_length # update the length counter

          # if it is the last sentence
            if count == len(sentences) - 1:
                chunks.append(chunk.strip()) # save the chunk

        else: 
            chunks.append(chunk.strip()) # save the chunk

            # reset 
            length = 0 
            chunk = ""

            # take care of the overflow sentence
            chunk += sentence + " "
            length = len(tokenizer.tokenize(sentence))

    inputs = [tokenizer(chunk, return_tensors="pt") for chunk in chunks]
    

    results = []
    for input in inputs:
        output = model.generate(**input)
        results += [tokenizer.decode(*output, skip_special_tokens=True)]

    final_str = "\n".join(results)

    return final_str


st = time.time()

summaries = []
for i in tqdm(range(len(wsj_final_df))):
    try:
        text_input = wsj_final_df.iloc[i]['text']
        text_summary = summarize_text(text_input)
        summaries += [text_summary]
    except:
        summaries += ['']
    
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

wsj_final_df['summary'] = summaries


# Convert to Speech

all_text = ''
for i in range(len(wsj_final_df)): 
    counter = i + 1
    title = wsj_final_df.iloc[i]['title']
    smry = wsj_final_df.iloc[i]['summary']
    temp_text = """
    Article number {}. 
    \n
    Title: {}.
    \n
    Summary: {}.
    \n
    """.format(counter, title, smry)
    
    
    all_text += temp_text.replace("\n"," ")
    
from gtts import gTTS
import os

def text_to_speech(text, output_filename, lang="en"):
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(output_filename)

output_file_mp3 = "wsj_summary_{}.mp3".format(date_today)

text_to_speech(all_text, output_file_mp3)


# convert to Video 
from moviepy.editor import AudioFileClip, ColorClip, CompositeVideoClip, TextClip

audio = AudioFileClip(output_file_mp3)
audio = audio.set_duration(audio.duration)
 

background = ColorClip((640, 360), color=(0, 0, 0), duration=audio.duration)
text = TextClip("{}".format(date_today), fontsize=70, color='white')
text = text.set_position(('center', 'center'))
video = CompositeVideoClip([background, text])

video = video.set_audio(audio)
video.duration = audio.duration

video.write_videofile('wsj_summary_{}.mp4'.format(date_today), codec='libx264', audio_codec='aac', fps=30)

output_video_file = 'wsj_summary_{}.mp4'.format(date_today)

# push to youtube (oauth2 required)
import os
import pickle
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
import googleapiclient.http
import google 

import os
import pickle
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import googleapiclient.discovery

def authenticate_youtube_api():
    scopes = ["https://www.googleapis.com/auth/youtube.upload"]
    api_service_name = "youtube"
    api_version = "v3"
    client_secrets_file = "your_crendetials.json"
    credentials_file = "credentials.pickle"

    credentials = None

    if os.path.exists(credentials_file):
        with open(credentials_file, 'rb') as token:
            credentials = pickle.load(token)

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(google.auth.transport.requests.Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, scopes)
            credentials = flow.run_local_server(port=8080)

        with open(credentials_file, 'wb') as token:
            pickle.dump(credentials, token)

    youtube = googleapiclient.discovery.build(api_service_name, api_version, credentials=credentials)

    return youtube


def upload_video(youtube, video_file, title, description, tags, category_id):
    request_body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": category_id
        },
        "status": {
            "privacyStatus": "unlisted"
        }
    }

    media_file = googleapiclient.http.MediaFileUpload(video_file, chunksize=-1, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=request_body, media_body=media_file)
    response = None

    while response is None:
        status, response = request.next_chunk()
        if response is not None:
            if "id" in response:
                print(f"Video ID: {response['id']} was successfully uploaded.")
            else:
                print("The upload failed with an unexpected response: %s" % response)


video_file = output_video_file
title = f"WSJ Summary {date_today}"
description = f"This is a summary of the Wall Street Journal on {date_today}"
tags = ["WSJ", "summary", "news"]
category_id = "25"  # News & Politics

youtube = authenticate_youtube_api()
upload_video(youtube, video_file, title, description, tags, category_id)
