function add_colors(hexagons, colors) {
    hexagons.style("fill", function (d,i) {
        return colors[i];
    });
}

function add_interaction(hexagons, tooltip, clusters, colors, on_cluster_clicked) {
        // reset opacity
        d3.selectAll(d3.selectAll(".hexagon")[0])
           .transition()
           .duration(0)
           .style("fill-opacity", 1);

        // reset tooltip
        tooltip.transition()
            .duration(0)
            .style("opacity", 0);



    var find_cluster = function(cell) {
            for (var i = 0; i < clusters.length; i++) {
                for (let cluster_cell of clusters[i].cell_ids) {
                    if (cell == cluster_cell) {
                        return i;
                    }
                }
            }
            return null;
        }

    var on_cell_clicked = function (d, i) {
         var cluster_id = find_cluster(i);
         if (cluster_id != null) {
             on_cluster_clicked(clusters[cluster_id].id);
         }

        document.getElementById("debug_view").innerHTML =
            `cell_id: "${i}" ` +
            `cluster_id: "${clusters[cluster_id].id}"` +
            `color: "${colors[i]}" ` +
            `label: "${clusters[cluster_id].label}" \n`
    }

    function on_mouse_over(d, i) {
        cluster_id = find_cluster(i);
        if (cluster_id == null) return;

        var f = d3.selectAll(".hexagon")[0].filter(function(d, j) {
            for (let cell_id of clusters[cluster_id].cell_ids) {
                if (cell_id  == j) {
                    return true;
                }
            }

            return false;
        });

        d3.selectAll(f)
            .transition()
            .duration(10)
            .style("fill-opacity", 0.3);

        d3.select(this).style("cursor", "pointer");

        tooltip.transition()
            .duration(200)
            .style("opacity", .9);

        tooltip.html(clusters[cluster_id].label)
            .style("left", (d3.event.pageX) + "px")
            .style("top", (d3.event.pageY - 28) + "px");
    }

     function on_mouse_out(d, i, zoom) {
        cluster_id = find_cluster(i);
        if (cluster_id == null) return;

        var f = d3.selectAll(".hexagon")[0].filter(function(d, j) {
            for (let cell_id of clusters[cluster_id].cell_ids) {
                if (cell_id == j) {
                    return true;
                }
            }

            return false;
        });
        d3.selectAll(f)
           .transition()
           .duration(0)
           .style("fill-opacity", 1);

        d3.select(this).style("cursor", "default");

        tooltip.transition()
            .duration(500)
            .style("opacity", 0);
    }

    hexagons
        .on("mouseover", on_mouse_over)
        .on("mouseout", on_mouse_out)
        .on("click", on_cell_clicked)
    ;
}

function add_clusters_border(svg, MapColumns, MapRows, clusters, truePoints, hexRadius) {

    ///////////////////////////////////////////////////////////////////////////
    ///// Function to calculate neighbouring cells from clusters           ////
    ///////////////////////////////////////////////////////////////////////////
    var to_ij = function(cell_id) {
        // cell id to row and column indexes
        row =  Math.floor(cell_id / MapColumns)
        column = cell_id - row * MapColumns
        return [row, column]
    }

    var to_id = function(i, j) {
        // cell indexes to id
        return i * MapColumns + j
    }


    var find_border = function(cluster1, cluster2) {
        var neighbours = function*(cell_id) {
            var ij = to_ij(cell_id)
            row = ij[0];
            column = ij[1];
            if (row % 2 == 0) {
                if (row - 1 >= 0) {
                    if (column - 1 >= 0) {
                        yield to_id(row-1, column-1);
                    }
                }
                yield to_id(row - 1, column);
                if (column - 1 >= 0) {
                    yield to_id(row, column - 1);
                }
                if (column + 1 < MapColumns) {
                    yield to_id(row, column + 1);
                }
                if (row + 1 < MapRows) {
                    if (column - 1 >= 0) {
                        yield to_id(row + 1, column - 1);
                    }
                    yield to_id(row + 1, column);
                }
            }
            else {
                if (row - 1 >= 0) {
                    if (column + 1 < MapColumns) {
                        yield to_id(row - 1, column + 1);
                    }
                }
                yield to_id(row - 1, column);
                if (column - 1 >= 0) {
                    yield to_id(row, column - 1)
                }
                if (column + 1 < MapColumns) {
                    yield to_id(row, column + 1);
                }
                if (row + 1 < MapRows) {
                    if (column + 1 < MapColumns) {
                        yield to_id(row + 1, column + 1);
                    }
                    yield to_id(row + 1, column);
                }
            }
        }


        var result = []
        for (let cell_id of cluster1) {
            for (let neighbour of neighbours(cell_id)) {
                if (cluster2.includes(neighbour)) {
                    result.push([cell_id, neighbour]);
                }
            }
        }

        return result
    }

    var find_neighbours = function(clusters) {
        var result = []
        for (var i = 0; i < clusters.length; i++) {
            cluster = clusters[i];
            for (var j = i + 1; j < clusters.length; j++) {
                let border_ij = find_border(cluster.cell_ids, clusters[j].cell_ids);
                for (let neighbour of border_ij) {
                    result.push(neighbour);
                }
            }
        }
        return result
    }

    let neighbour = find_neighbours(clusters);

    ///////////////////////////////////////////////////////////////////////////
    ///// Function to calculate the line segments between two node numbers ////
    ///////////////////////////////////////////////////////////////////////////


    //Initiate some variables
    var Sqr3 = 1/Math.sqrt(3);
    var lineData = [];
    var Node1,
        Node2,
        Node1_xy,
        Node2_xy,
        P1,
        P2;

    //Calculate the x1, y1, x2, y2 of each line segment between neighbours
    for (var i = 0; i < neighbour.length; i++) {
        Node1 = neighbour[i][0];
        Node2 = neighbour[i][1];

        //An offset needs to be applied if the node is in an uneven row
        if (Math.floor(Math.floor((Node1/MapColumns)%2)) != 0) {
            Node1_xy = [(truePoints[Node1][0]+(hexRadius/(Sqr3*2))),truePoints[Node1][1]];
        }
        else {
            Node1_xy = [truePoints[Node1][0],truePoints[Node1][1]];
        }

        //An offset needs to be applied if the node is in an uneven row
        if (Math.floor(Math.floor((Node2/MapColumns)%2)) != 0) {
            Node2_xy = [(truePoints[Node2][0]+(hexRadius/(Sqr3*2))),truePoints[Node2][1]];
        }
        else {
            Node2_xy = [truePoints[Node2][0],truePoints[Node2][1]];
        }//else

        //P2 is the exact center location between two nodes
        P2 = [(Node1_xy[0]+Node2_xy[0])/2,(Node1_xy[1]+Node2_xy[1])/2]; //[x2,y2]
        P1 = Node1_xy; //[x1,x2]

        //A line segment will be drawn between the following two coordinates
        lineData.push([(P2[0] + Sqr3*(P1[1] - P2[1])),(P2[1] + Sqr3*(P2[0] - P1[0]))]); //[x3_top, y3_top]
        lineData.push([(P2[0] + Sqr3*(P2[1] - P1[1])),(P2[1] + Sqr3*(P1[0] - P2[0]))]); //[x3_bottom, y3_bottom]
    }//for i

    ///////////////////////////////////////////////////////////////////////////
    /////////////////// Draw the black line segments //////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    var lineFunction = d3.svg.line()
              .x(function(d) {return d[0];})
              .y(function(d) {return d[1];})
              .interpolate("linear");


    d3.selectAll("path.border").remove();

    //Loop over the linedata and draw each line
    for (var i = 0; i < lineData.length; i+=2) {
        svg.append("path")
            .attr("class", "border")
            .attr("d", lineFunction([lineData[i],lineData[i+1]]))
            .attr("stroke", "black")
            .attr("stroke-width", 2)
            .attr("fill", "none");
    }
}


function plot_som(MapColumns, MapRows, data, on_cluster_clicked) {


    ///////////////////////////////////////////////////////////////////////////
    ////////////// Initiate SVG and create hexagon centers ////////////////////
    ///////////////////////////////////////////////////////////////////////////

    var width = 310 * 2;
    var height = 250 * 2;

    //svg sizes and margins
    var margin = {
        top: 30,
        right: 20,
        bottom: 20,
        left: 20
    };

    //The maximum radius the hexagons can have to still fit the screen
    var hexRadius = d3.min([width/((MapColumns + 0.5) * Math.sqrt(3)),
                height/((MapRows + 1. / 3.) * 1.5)]);

    //Set the new height and width of the SVG based on the max possible
    width = MapColumns * hexRadius * Math.sqrt(3);
    heigth = MapRows * 1.5 * hexRadius + 0.5 * hexRadius;

    //Set the hexagon radius
    var hexbin = d3.hexbin().radius(hexRadius);

    //Calculate the center positions of each hexagon
    var points = [];
    for (var i = 0; i < MapRows; i++) {
        for (var j = 0; j < MapColumns; j++) {
            points.push([hexRadius * j * Math.sqrt(3), hexRadius * i * 1.5]);
        }
    }

    d3.select("svg").remove();

    //Create SVG element
    var svg = d3.select("#som_view").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .on("dblclick.zoom", null) // disable zoom on doubleclick
        ;


    var tooltip = d3.select("#som_view").append("div")
        .attr("class", "tooltip")
        .style("opacity", 0);


    ///////////////////////////////////////////////////////////////////////////
    ////////////////////// Draw hexagons ////////////// ///////////////////////
    ///////////////////////////////////////////////////////////////////////////

    //Start drawing the hexagons
    var hexagons = svg.append("g")
        .selectAll(".hexagon")
        .data(hexbin(points))
        .enter().append("path")
        .attr("class", "hexagon")
        .attr("d", function (d) {
            return "M" + d.x + "," + d.y + hexbin.hexagon();
        })
        .attr("stroke", function (d,i) {
            return "#fff";
        })
        .attr("stroke-width", "1px");



    on_zoom_changed = function(zoom_level) {
        add_clusters_border(
            svg, MapColumns, MapRows, data[zoom_level].clusters, points, hexRadius);

        add_colors(hexagons, data[zoom_level].colors);

        add_interaction(
            hexagons, tooltip, data[zoom_level].clusters,
            data[zoom_level].colors, on_cluster_clicked
        );
    };

    on_zoom_changed(0);

    var zoom = d3.behavior.zoom()
            .on("zoom", function () {
                let scale = d3.event.scale;
                let zoom = scale.toString().substring(0, 3);
                document.getElementById("zoom_view").innerHTML =`zoom = "${zoom}" `
            })
            .on("zoomend", function() {
                let scale = zoom.scale() - 1;
                let zoom_level = Math.floor(scale);
                on_zoom_changed(zoom_level);
            })
            .scaleExtent([1, data.length]);

    svg.call(zoom);
}

$(document).ready(function() {

    // init the socket and namespace(optional)
    namespace = '/';
    let connection_str = location.protocol + '//' + document.domain + ':' + location.port + namespace;
    var socket = io.connect(connection_str);

    socket.on('connect', function() {
        socket.emit('on_connected', null);
    });

    var on_cluster_clicked = function (id) {
        socket.emit('on_cluster_clicked', id);
        return false;
    }

    socket.on('som_config', function(som_config) {
        var config = JSON.parse(som_config);
        plot_som(
            config.map_columns,
            config.map_rows,
            config.data,
            on_cluster_clicked
        );
    });
});
