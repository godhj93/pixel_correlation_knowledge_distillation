import tensorflow as tf

def pixel_correlation_loss(y_t,y_s):
    
    B,H,W,C = y_t.shape
    y_t_flattend = tf.reshape(y_t, (B,-1,C))
    y_s_flattend = tf.reshape(y_s, (B,-1,C))
    
    loss_pc = 0
    for i in range(H):
        for j in range(W):
            pt_i, pt_j = y_t_flattend[:,i,:], y_t_flattend[:,j,:]
            ps_i, ps_j = y_s_flattend[:,i,:], y_s_flattend[:,j,:]
            loss_pc += (similarity(pt_i, pt_j) - similarity(ps_i, ps_j))**2
    
    return tf.reduce_sum(loss_pc/H/W)

def similarity(p_i, p_j):
    
    return tf.matmul(tf.transpose(p_i),p_j)/(tf.norm(p_i,ord=2)*tf.norm(p_j,ord=2))    
    
