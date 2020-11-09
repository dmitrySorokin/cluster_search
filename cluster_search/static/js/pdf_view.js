    $(document).ready(function() {

        // init the socket and namespace(optional)
        namespace = '/';
        let connection_str = location.protocol + '//' + document.domain + ':' + location.port + namespace;
        var socket = io.connect(connection_str);

        socket.on('pdf_links', function(data) {
            var pdf_links = document.getElementById("pdf_links");
            pdf_links.innerHTML = "";
            for (let element of data) {
                let pdf_item = JSON.parse(element);

                var div = document.createElement('div');
                div.style.padding = "5px";

                var a = document.createElement('a');
                a.href = connection_str + 'open_pdf/' +
                    '?id_file=' + pdf_item.id_file +
                    '&page=' + pdf_item.page;
                a.textContent = pdf_item.name;
                div.appendChild(a);

                div.appendChild(document.createElement('br'));

                var page = document.createElement('div');
                var page_text = document.createTextNode('страница: ' + pdf_item.page);
                page.appendChild(page_text);
                page.style.color = "green";
                div.appendChild(page);

                let parag = document.createTextNode(pdf_item.parag);
                div.appendChild(parag);

                div.appendChild(document.createElement('br'));

                pdf_links.appendChild(div);
            }

            document.getElementById("pdf_list_view").scrollTop = 0;
        });

    });
