from flask import Flask, Response
import dash
from dash.dependencies import Output,  Input
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import dash_table
import sqlite3
from utilities.centroidtracker import CentroidTracker
from utilities.trackableobject import TrackableObject
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import dlib
import cv2
import datetime
import db
import plotly.express as px
import pandas as pd
import webbrowser
from threading import Timer


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--camera", default=0,
                help="use external camera")
ap.add_argument("-i", "--input", type=str,
                help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
                help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=15,
                help="# of skip frames between detections")
args = vars(ap.parse_args())


# df = pd.DataFrame({'Date': [], 'Number of People':[] })
# df.to_csv('Output.csv')

# if a video path was not supplied, use the webcam
if not args.get("input", False):
    print("[INFO] Streaming Video from Camera...")
    vs = cv2.VideoCapture(int(args["camera"]))

# get a reference to the video file
else:
    print("[INFO] Opening video file...")
    vs = cv2.VideoCapture(args["input"])


def gen(vs):

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    # load the serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(
        'mobilenet_ssd/MobileNetSSD_deploy.prototxt', 'mobilenet_ssd/MobileNetSSD_deploy.caffemodel')

    # initialize the video writer, it'll be instantiated later
    writer = None

    # initialize the frame dimensions (it'll be set using
    # the first frame from the video or webcam)
    W = None
    H = None

    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    # initialize the total number of frames processed thus far, along
    # with the total number of people entering or leaving
    totalFrames = 0
    totalEntering = 0
    totalLeaving = 0

    # start the frames per second throughput estimator
    # fps = FPS().start()

    while True:
        # get the next frame and handle if we are reading from either
        # VideoCapture or VideoStream
        _, frame = vs.read()
        frame = frame if args.get("input", False) else frame

        # if we are viewing a video and we did not grab a frame then we
        # have reached the end of the video
        if args["input"] is not None and frame is None:
            break

        # resize the frame to have a maximum width of 520 pixels
        # then convert the frame from BGR to RGB for dlib
        frame = imutils.resize(frame, width=520)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # if we are supposed to be writing a video to disk, initialize
        # the writer
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                     (W, H), True)

        # initialize the current status along with our list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        status = "Waiting"
        rects = []

        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if totalFrames % args["skip_frames"] == 0:
            # set the status and initialize our new set of object trackers
            status = "Detecting"
            trackers = []

            # convert the frame to a blob and pass the blob through the
            # network and obtain the detections
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by requiring a minimum
                # confidence
                if confidence > args["confidence"]:

                    temp_confidence = confidence

                    # extract the index of the class label from the
                    # detections list
                    idx = int(detections[0, 0, i, 1])

                    # if the class label is not a person, ignore it
                    if CLASSES[idx] != "person":
                        continue

                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    # construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation
                    # tracker
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    trackers.append(tracker)

        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        else:
            # loop over the trackers
            for tracker in trackers:
                # set the status of our system to be 'tracking' rather
                # than 'waiting' or 'detecting'
                status = "Tracking"

                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))

        # draw a horizontal line in the center of the frame -- once an
        # object crosses this line we will determine whether they were
        # moving 'up' or 'down'
        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                # check to see if the object has been counted or not
                if not to.counted:
                    # if the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object
                    if direction < 0 and centroid[1] < H // 2:
                        totalLeaving += 1
                        to.counted = True

                        timestamp = str(
                            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                        db.data_entry(totalEntering, totalLeaving, timestamp)
                        # db.read_from_db()

                        # print(f'Total Leaving {totalLeaving} {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif direction > 0 and centroid[1] > H // 2:
                        totalEntering += 1
                        to.counted = True

                        timestamp = str(
                            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        db.data_entry(totalEntering, totalLeaving, timestamp)
                        # db.read_from_db()
                        # print(f'Total Entering {totalEntering} {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "Person {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # construct a tuple of information we will be displaying on the
        # frame
        info = [
            ("Leaving", totalLeaving),
            ("Entering", totalEntering),
            ("Status", status),
            ("Total people inside", 0 if totalEntering -
             totalLeaving < 0 else totalEntering-totalLeaving)
        ]

        # loop over the info tuples and draw them on our frame
        # for (i, (k, v)) in enumerate(info):
        #     text = "{}: {}".format(k, v)
        #     cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 0), 2)

        # check to see if we should write the frame to disk
        if writer is not None:
            writer.write(frame)

        # show the output frame
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        # frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        totalFrames += 1

    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()

    # if we are not using a video file, stop the camera video stream
    if not args.get("input", False):
        vs.stop()

    # otherwise, release the video file pointer
    else:
        vs.release()


# Instantiate the server
server = Flask(__name__)
app = dash.Dash(__name__, server=server)


@server.route('/video_feed')
def video_feed():
    return Response(gen(vs),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Writing page Layout
app.layout = html.Div(
    className='bg-white w-full py-4 px-4 ',

    children=[

        html.Div(

            className=' bg-white mx-auto shadow-2xl pb-24 pt-6',

            children=[

                html.Div(
                    # Main header
                    children=[
                        html.H5('People Counter Dashboard', id='main-header',
                                className='uppercase tracking-wide text-gray-800 font-bold text-center text-4xl py-6'),
                        html.Hr(className='divide-y-4 mb-8')
                    ]

                ),


                html.Div(
                    className='flex items-center justify-evenly mb-4 px-3',

                    children=[

                        # Video
                        html.Div(

                            html.Img(src="/video_feed",
                                     className='object-center rounded-lg object-cover object-center overflow-hidden w-full h-96'),

                            className='shadow-lg rounded-lg  w-1/2 mr-3'),


                        # Score card main div
                        html.Div(

                            className='shadow-lg rounded-lg flex justify-between items-center px-4 py-32 w-3/6',

                            children=[

                                # Score card number 1
                                html.Div(
                                    className='flex flex-col px-10',

                                    children=[
                                        html.H1(
                                            id='people-entering', className='text-8xl font-bold py-2'),
                                        html.H1(
                                            'Entering', className='text-sm font-bold')
                                    ],

                                ),

                                # score card number 2
                                html.Div(
                                    className='flex flex-col px-10',

                                    children=[
                                        html.H1(
                                            id='people-inside', className='text-8xl font-bold py-2'),
                                        html.H1(
                                            'Inside', className='text-sm font-bold')
                                    ],

                                ),

                                # Scorecard number 3
                                html.Div(
                                    className='flex flex-col px-10',

                                    children=[
                                        html.H1(
                                            id='people-leaving', className='text-8xl font-bold py-2'),
                                        html.H1(
                                            'Exiting', className='text-sm font-bold')
                                    ],

                                ),



                            ],


                        )

                    ]
                ),


                html.Div(
                    className='flex items-center justify-evenly mb-4',

                    children=[

                        html.Div(


                            dcc.Graph(id='live-graph',
                                      animate=True), className='shadow-lg rounded-lg'),

                        html.Div(
                            dcc.Graph(id='live-bar-graph',
                                      animate=True,
                                      responsive=True,
                                      ),
                            className='shadow-lg rounded-lg'
                        )
                    ]

                ),

                html.Div(

                    children=[
                        html.Div(
                            className='shadow-lg rounded-lg',
                            children=[
                                dash_table.DataTable(
                                    id='data-table',
                                    columns=[
                                        {'name': 'Date',
                                         'id': 'Date'},
                                        {"name": 'Total Visits',
                                            "id": 'People_Visits'},


                                    ],
                                    page_size=10,
                                    style_cell={'textAlign': 'center'},
                                    style_header={
                                        'fontSize': '20px',
                                        'fontWeight': 'bold'
                                    },

                                ),

                                html.Div(
                                    [html.Button("Download CSV",
                                                 id="btn_csv",
                                                 className='py-4 px-4 bg-blue-500 mx-4 my-4 text-white rounded-lg shadow-lg'),
                                     dcc.Download(
                                        id="download-csv")]
                                )
                            ]
                        )
                    ],

                    className='px-4 py-4',
                ),



                dcc.Interval(
                    id='graph-update',
                    interval=1*1000,
                    n_intervals=0
                ),
            ])])


# Function To read from database
def read_from_db(dbase, db):

    dbase.execute("SELECT total_entered, total_left, datestamp FROM people")
    data = dbase.fetchall()

    totalEntered_list = []
    totalLeft_list = []
    timestamp_list = []

    for tuple in data:
        totalE = tuple[0]
        totalL = tuple[1]
        timestamp = tuple[2]

        totalEntered_list.append(totalE)
        totalLeft_list.append(totalL)
        timestamp_list.append(timestamp)

    dbase.close()
    db.close()

    return totalEntered_list, totalLeft_list, timestamp_list


@ app.callback(Output('live-graph', 'figure'),
               [Input('graph-update', 'n_intervals')])
def update_graph_scatter(n):

    db = sqlite3.connect(
        'people.db', detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    dbase = db.cursor()

    totalEntered_list, totalLeft_list, timestamp_list = read_from_db(dbase, db)

    totalEntered_list = np.array(totalEntered_list)
    totalLeft_list = np.array(totalLeft_list)

    number_of_people_inside = totalEntered_list-totalLeft_list

    data = go.Scatter(
        x=timestamp_list,
        y=number_of_people_inside,
        name='Scatter',
        mode='lines+markers'
    )

    layout = go.Layout(
        title="",
        autosize=True,

    )

    # data = px.area(
    #     x=timestamp_list,
    #     y=number_of_people_inside,
    #     line_shape='spline',
    #     labels={'x': 'Timestamp', 'y': 'Number of persons inside'}

    # )

    # layout = go.Layout(
    #     title="Hame",
    #     autosize=True,
    # )

    figure = go.Figure(data=data, layout=layout)

    return figure


@ app.callback(Output('live-bar-graph', 'figure'),
               [Input('graph-update', 'n_intervals')])
def update_graph_scatter(n):

    db = sqlite3.connect(
        'people.db', detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    dbase = db.cursor()

    totalEntered_list, totalLeft_list, timestamp_list = read_from_db(dbase, db)

    totalEntered_list = np.array(totalEntered_list)
    totalLeft_list = np.array(totalLeft_list)

    data = go.Bar(
        x=['People Entering', 'People Leaving'],
        y=[totalEntered_list[-1], totalLeft_list[-1]],
        marker=dict(color=['#0D65D9', '#72B7B2'],
                    colorscale='viridis'),
    )

    layout = go.Layout(
        title="",
        autosize=True,
        yaxis=dict(type='linear',
                   range=[0, int(max(totalEntered_list[-1], totalLeft_list[-1])+3)])

    )

    figure = go.Figure(data=data, layout=layout)

    return figure


@ app.callback(Output('people-entering', 'children'),
               [Input('graph-update', 'n_intervals')])
def update_graph_scatter(n):

    db = sqlite3.connect(
        'people.db', detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    dbase = db.cursor()

    totalEntered_list, totalLeft_list, timestamp_list = read_from_db(dbase, db)

    totalEntered_list = np.array(totalEntered_list)
    totalLeft_list = np.array(totalLeft_list)

    return f'{int(totalEntered_list[-1])}'


@ app.callback(Output('people-inside', 'children'),
               [Input('graph-update', 'n_intervals')])
def update_graph_scatter(n):

    db = sqlite3.connect(
        'people.db', detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    dbase = db.cursor()

    totalEntered_list, totalLeft_list, timestamp_list = read_from_db(dbase, db)

    totalEntered_list = np.array(totalEntered_list)
    totalLeft_list = np.array(totalLeft_list)

    return f'{0 if int(totalEntered_list[-1] - totalLeft_list[-1]) < 0 else int(totalEntered_list[-1] - totalLeft_list[-1])}'


@ app.callback(Output('people-leaving', 'children'),
               [Input('graph-update', 'n_intervals')])
def update_graph_scatter(n):

    db = sqlite3.connect(
        'people.db', detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    dbase = db.cursor()

    totalEntered_list, totalLeft_list, timestamp_list = read_from_db(dbase, db)

    totalEntered_list = np.array(totalEntered_list)
    totalLeft_list = np.array(totalLeft_list)

    return f'{int(totalLeft_list[-1])}'


@ app.callback(Output('data-table', 'data'),
               [Input('graph-update', 'n_intervals')])
def update_graph_scatter(n):

    db = sqlite3.connect(
        'people.db', detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    dbase = db.cursor()

    totalEntered_list, totalLeft_list, timestamp_list = read_from_db(dbase, db)

    temp = [datetime.datetime.strptime(
        date, "%Y-%m-%d %H:%M:%S") for date in timestamp_list]

    df = pd.DataFrame({
        'People_Entering': totalEntered_list,
        # 'People_Leaving': totalLeft_list,
        'Date': [f'{str(date.year)}-{str(date.month)}-{str(date.day)}' for date in temp]
    },
    )

    unique = df.Date.unique().tolist()

    dic = {}
    for i in unique:
        dic[i] = df.where(df.Date == i).People_Entering.dropna().max()

    dates = []
    people_visits = []

    for key, value in dic.items():
        dates.append(key)
        people_visits.append(value)

    table_df = pd.DataFrame(
        {'Date': dates, 'People_Visits': people_visits}
    )

    return table_df.to_dict('records')


@app.callback(
    Output("download-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):

    db = sqlite3.connect(
        'people.db', detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    dbase = db.cursor()

    totalEntered_list, totalLeft_list, timestamp_list = read_from_db(dbase, db)

    df = pd.DataFrame({
        'Timestamp': timestamp_list,
        'People_Entering': totalEntered_list,
        'People_Leaving': totalLeft_list}
    )

    # df.to_csv('Output.csv')
    # df = pd.read_csv('./Output.csv')

    return dcc.send_data_frame(df.to_csv, "data.csv")


def openBrowser():
    webbrowser.open_new_tab('http://127.0.0.1:4500/')


if __name__ == '__main__':
    Timer(2, openBrowser).start()
    app.run_server(port=4500)
