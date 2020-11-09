from flask import Flask
from flask_socketio import SocketIO

from .configs import Config
from .core import ClusterSearchApi


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    api = ClusterSearchApi()

    from .routes import  bind_all_routes
    bind_all_routes(app, api)

    sio = SocketIO(app)
    from .events import bind_all_events
    bind_all_events(sio, api)

    return app, sio

def main():
    app, sio =  create_app() 
    sio.run(app, debug = Config.debug, host='127.0.0.1', port='5555')
