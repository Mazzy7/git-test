import tensorflow as tf

# freeze tensorflow model
saver = tf.train.import_meta_graph('model-300.meta')
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def() # print 
sess = tf.Session()
saver.restore(sess, "model-300")
#output_node_names="input_y"
#output_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names.split(",")  )
#for op in graph.get_operations():
#    print(op.name)
#output_node_names = []
#output_node_names.append("output/predictions")  
#output_graph_def = tf.graph_util.convert_variables_to_constants(
#    sess,
#    input_graph_def,
#    output_node_names
#)
output_node_names = []
output_node_names.append("output/predictions")  
output_graph_def = tf.graph_util.convert_variables_to_constants(
    sess,
    input_graph_def,
    output_node_names
)
#for op in graph.get_operations():
#    print(op.name)
output_graph="pos-neg-model1.pb"
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
 
sess.close()