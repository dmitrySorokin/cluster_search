from flask import request, render_template, redirect


def bind_all_routes(app, api):

    @app.route('/')
    @app.route('/index')
    def index():
        print('\n\nrequest method: ', request.method)
        return render_template('index.html')

    @app.route('/open_pdf/')
    def open_pdf():
        id_file = request.args.get('id_file', default=None, type=int)
        page = request.args.get('page', default=1, type=int)

        print('try to open "{}" at page "{}"'.format(id_file, page))

        path = api.path_to_file(id_file)
        return redirect('/static/js/external/pdfjs/web/viewer.html?'
                        'file={}#'
                        'page={}'.format(path, page))

