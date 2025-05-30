from flask import Flask, render_template, Response, jsonify, redirect
from camera import VideoCamera, music_rec, get_current_video  # ✅ Correct import

app = Flask(__name__)

headings = ("Name", "Album", "Artist")
df1 = music_rec()
df1 = df1.head(15)

@app.route('/')
def index():
    print(df1.to_json(orient='records'))
    return render_template('index.html', headings=headings, data=df1)

def gen(camera):
    while True:
        global df1
        frame, df1 = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/t')
def gen_table():
    return df1.to_json(orient='records')

# ✅ Fix: Correct video route using imported get_current_video
@app.route('/current_video')
def current_video():
    video_id = get_current_video()
    return jsonify({'video_id': video_id})

# ✅ Route to open YouTube song (new tab)
@app.route('/play_song')
def play_song():
    video_id = get_current_video()
    url = f"https://www.youtube.com/watch?v={video_id}"
    return redirect(url)

if __name__ == '__main__':
    app.debug = True
    app.run()
