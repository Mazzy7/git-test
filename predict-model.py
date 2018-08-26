import tensorflow as tf
import argparse

frozen_graph="pos-neg-model1.pb"

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

if __name__ == '__main__':


    # We use our "load_graph" function
    graph = load_graph(frozen_graph)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)

#    # We access the input and output nodes 
    x = graph.get_tensor_by_name('prefix/input_x:0')
    y = graph.get_tensor_by_name('prefix/output/predictions/dimension:0')

#
#    # We launch a Session

    with tf.Session(graph=graph) as sess:

        y_out = sess.run(y, feed_dict={
            x: [[4719,   59,  182,   34,  190,  804,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0]] 
        })

#
        print(y_out) 
#        