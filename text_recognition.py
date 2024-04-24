import whisper
from datetime import timedelta

model = whisper.load_model('base')

def sec_to_hour_min_sec(seconds):
    dt = timedelta(seconds=seconds)
    hours, remainder = divmod(dt.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{}:{:02d}:{:02d}'.format(hours, minutes, seconds)


async def async_text_recognition(path):
    res = model.transcribe(path)
    txt = []
        
    for seg in res['segments']:
        basic_txt = seg['text']
        # txt_to_trans = seg['text']
        # if trans.detect(txt_to_trans).lang == 'ko':
        #     print('[{0} ~ ] {1}'.format(sec_to_hour_min_sec(seg['start']), txt_to_trans))
        #     continue
        
        # trans_text = trans.translate(seg['text'], dest='ko').text
        
        try:
            # 중복 확인
            if basic_txt == basic_txt2:
                continue
            else:
                print('[{0} ~ ] {1}'.format(sec_to_hour_min_sec(seg['start']), 
                                        basic_txt))
        except:
            print('[{0} ~ ] {1}'.format(sec_to_hour_min_sec(seg['start']), 
                                        basic_txt))
        basic_txt2 = basic_txt
        txt.append(str('[{0} ~ ] {1}'.format(sec_to_hour_min_sec(seg['start']), 
                                             basic_txt)))
    
    return txt

