import jax
import jax.numpy as jnp


@jax.jit
def BCE_loss(labels, x) : 
    num_classes = x.shape[1]
    
    return -jnp.mean(jnp.sum(jax.nn.one_hot(labels, num_classes)*jnp.log(x), axis = -1))

@jax.jit
def MSE_loss(x, y): 
    return jnp.mean((x - y)**2)

@jax.jit
def accuracy(labels, pred) : 
#     pred_labels = jnp.abs(pred - labels) 
#     accuracy = jnp.sum(pred_labels < 1)/len(pred) 
    
    accuracy = jnp.sum(jnp.argmax(pred, axis = 1) == labels)/len(pred)
    
    
    return accuracy 

def get_metrics(loss_type):
    
    switcher = {
        "MSE_loss": MSE_loss,
        "BCE_loss": BCE_loss,
        "accuracy" : accuracy
    }
    loss = switcher.get(loss_type, lambda: None)
    if loss is None:
        raise TypeError("Specified loss does not exist!")

    return loss