from flask_socketio import emit


def bind_all_events(sio, api):

    @sio.on('on_connected', namespace='/')
    def on_connected(arg):
        emit('som_config', api.som_config())

    @sio.on('on_cluster_clicked', namespace='/')
    def on_cluster_clicked(cluster_id):
        emit('pdf_links', api.pdf_links(cluster_id))
