import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import gym
env = gym.make("CartPole-v0")
# 1. Specify the neural network architecture
n_inputs = 4 # == env.observation_space.shape[0]
n_hidden = 4 # it's a simple task, we don't need more hidden neurons
n_outputs = 1 # only outputs the probability of accelerating left
initializer = tf.contrib.layers.variance_scaling_initializer()
# 2. Build the neural network
X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = fully_connected(X, n_hidden, activation_fn=tf.nn.elu,weights_initializer=initializer)
logits = fully_connected(hidden, n_outputs, activation_fn=None,weights_initializer=initializer)
outputs = tf.nn.sigmoid(logits)
# 3. Select a random action based on the estimated probabilities
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)


y = 1. - tf.to_float(action)
learning_rate = 0.01
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, variable in grads_and_vars]

gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))
training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
import numpy as np

def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards
def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards,discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]

# print(discount_rewards([10, 0, -50], discount_rate=0.8))
# print(discount_and_normalize_rewards([[10, 0, -50], [10, 20]], discount_rate=0.8))

n_iterations = 250 # number of training iterations
n_max_steps = 1000 # max steps per episode
n_games_per_update = 10 # train the policy every 10 episodes
save_iterations = 10 # save the model every 10 training iterations
discount_rate = 0.95
# with tf.Session() as sess:
#     init.run()
#     for iteration in range(n_iterations):
#         all_rewards = [] # all sequences of raw rewards for each episode
#         all_gradients = [] # gradients saved at each step of each episode
#         for game in range(n_games_per_update):
#             current_rewards = [] # all raw rewards from the current episode
#             current_gradients = [] # all gradients from the current episode
#             obs = env.reset()
#             for step in range(n_max_steps):
#                 action_val, gradients_val = sess.run([action, gradients],feed_dict={X: obs.reshape(1, n_inputs)}) # one obs
#                 obs, reward, done, info = env.step(action_val[0][0])
#                 current_rewards.append(reward)
#                 current_gradients.append(gradients_val)
#                 if done:
#                     break
#             all_rewards.append(current_rewards)
#             all_gradients.append(current_gradients)
#     # At this point we have run the policy for 10 episodes, and we are
#     # ready for a policy update using the algorithm described earlier.
#         all_rewards = discount_and_normalize_rewards(all_rewards,discount_rate=discount_rate)
#         feed_dict = {}
#         for var_index, grad_placeholder in enumerate(gradient_placeholders):
#         # multiply the gradients by the action scores, and compute the mean
#             mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index] for game_index, rewards in enumerate(all_rewards)
#                                       for step, reward in enumerate(rewards)],axis=0)
#             feed_dict[grad_placeholder] = mean_gradients
#         sess.run(training_op, feed_dict=feed_dict)
#         if iteration % save_iterations == 0:
#             saver.save(sess, "./my_policy_net_pg.ckpt")

env.close()
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

try:
    from pyglet.gl import gl_info
    openai_cart_pole_rendering = True   # no problem, let's use OpenAI gym's rendering function
except Exception:
    openai_cart_pole_rendering = False
def render_cart_pole(env, obs):
    if openai_cart_pole_rendering:
        # use OpenAI gym's rendering function
        return env.render(mode="rgb_array")
    else:
        # rendering for the cart pole environment (in case OpenAI gym can't do it)
        img_w = 600
        img_h = 400
        cart_w = img_w // 12
        cart_h = img_h // 15
        pole_len = img_h // 3.5
        pole_w = img_w // 80 + 1
        x_width = 2
        max_ang = 0.2
        bg_col = (255, 255, 255)
        cart_col = 0x000000 # Blue Green Red
        pole_col = 0x669acc # Blue Green Red

        pos, vel, ang, ang_vel = obs
        img = Image.new('RGB', (img_w, img_h), bg_col)
        draw = ImageDraw.Draw(img)
        cart_x = pos * img_w // x_width + img_w // x_width
        cart_y = img_h * 95 // 100
        top_pole_x = cart_x + pole_len * np.sin(ang)
        top_pole_y = cart_y - cart_h // 2 - pole_len * np.cos(ang)
        draw.line((0, cart_y, img_w, cart_y), fill=0)
        draw.rectangle((cart_x - cart_w // 2, cart_y - cart_h // 2, cart_x + cart_w // 2, cart_y + cart_h // 2), fill=cart_col) # draw cart
        draw.line((cart_x, cart_y - cart_h // 2, top_pole_x, top_pole_y), fill=pole_col, width=pole_w) # draw pole
        return np.array(img)
def render_policy_net(model_path, action, X, n_max_steps = 1000):
    frames = []
    env = gym.make("CartPole-v0")
    obs = env.reset()
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        for step in range(n_max_steps):
            img = render_cart_pole(env, obs)
            frames.append(img)
            action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
            obs, reward, done, info = env.step(action_val[0][0])
            if done:
                break
    env.close()
    return frames
def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,
import matplotlib.animation as animation
def plot_animation(frames, repeat=False, interval=40):
    plt.close()  # or else nbagg sometimes plots in the previous cell
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    return animation.FuncAnimation(fig, update_scene, fargs=(frames, patch), frames=len(frames), repeat=repeat, interval=interval)
frames = render_policy_net("./my_policy_net_pg.ckpt", action, X, n_max_steps=1000)
video = plot_animation(frames)
plt.show()