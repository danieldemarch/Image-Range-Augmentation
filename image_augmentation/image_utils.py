import numpy as np
from numba import jit
import tensorflow as tf
import random

def corrupt_images(inmages, inndexlist):
  outmages = inmages.copy()
  for inndex in inndexlist:
    rangewun = inndex*(256/8)
    rangetoo = (inndex+1)*(256/8)
    the_mask = np.logical_and(outmages >= rangewun, outmages < rangetoo)
    outmages[the_mask] = 0

  return outmages

def purify_images(corruption, pure, restore_index_list):
  outmages = corruption.copy()

  for inndex in restore_index_list:
    rangewun = inndex*(256/8)
    rangetoo = (inndex+1)*(256/8)

    the_mask = np.logical_and(pure >= rangewun, pure < rangetoo)

    outmages[the_mask] = pure[the_mask]

  return outmages

def obscure_top_to_bottom(pure_images, percent):
    outmages = pure_images.copy()
    
    rowstoremove = np.rint(outmages.shape[1]*percent)
        
    top_remove = np.random.randint(0, rowstoremove+1)
    
    bottom_remove = rowstoremove-top_remove
        
    outmages[:,0:top_remove,:,:] = 0
    
    outmages[:,int(outmages.shape[1]-bottom_remove):int(outmages.shape[1]),:,:] = 0
    
    return outmages, np.array([top_remove, int(outmages.shape[1]-bottom_remove)])

def obscure_left_to_right(pure_images, percent):
    outmages = pure_images.copy()
    
    colstoremove = np.rint(outmages.shape[1]*percent)
    
    left_remove = np.random.randint(0, colstoremove+1)
    
    right_remove = colstoremove-left_remove

    outmages[:,:,0:left_remove,:] = 0
    
    outmages[:,:,int(outmages.shape[1]-right_remove):int(outmages.shape[1]),:] = 0
    
    return outmages, np.array([left_remove, int(outmages.shape[1]-right_remove)])

def obscure_block(pure_images, blockdim, numblocks):
    outmages = pure_images.copy()
    
    maxdim = pure_images.shape[1]-blockdim
        
    xpos = np.random.randint(0, maxdim+1, (numblocks))
    ypos = np.random.randint(0, maxdim+1, (numblocks))
    
    xpos = xpos[xpos.argsort()]
    ypos = ypos[xpos.argsort()]
                    
    for i in range(len(xpos)):
        outmages[:,xpos[i]:(xpos[i]+blockdim),ypos[i]:(ypos[i]+blockdim),:] = 0
    
    return outmages, np.array([xpos, ypos])

def restore_top_to_bottom(pure_images, occluded_images, topmax, botmin, topamount, botamount):
    outmages = occluded_images.copy()
    
    toprange = np.array([topmax-topamount, topmax])
    
    botrange = np.array([botmin, botmin+botamount])
        
    outmages[:,toprange[0]:toprange[1],:,:] = pure_images[:,toprange[0]:toprange[1],:,:]
    outmages[:,botrange[0]:botrange[1],:,:] = pure_images[:,botrange[0]:botrange[1],:,:]
    
    return outmages

def restore_left_to_right(pure_images, occluded_images, leftindex, rightindex, leftamount, rightamount):
    outmages = occluded_images.copy()
    
    leftrange = np.array([leftindex-leftamount, leftindex])
    
    rightrange = np.array([rightindex, rightindex+rightamount])
    
    outmages[:,:,leftrange[0]:leftrange[1],:] = pure_images[:,:,leftrange[0]:leftrange[1],:]
    outmages[:,:,rightrange[0]:rightrange[1],:] = pure_images[:,:,rightrange[0]:rightrange[1],:]
    
    return outmages

def restore_block(pure_images, occluded_images, blockx, blocky, blockdim):
    
    outmages = occluded_images.copy()
    
    outmages[blockx:(blockx+blockdim),blocky:(blocky+blockdim),:] = pure_images[blockx:(blockx+blockdim),blocky:(blocky+blockdim),:]

    return outmages

def gen_lr_data(pure_images, percent, batch_size):
    
    lr_images = np.zeros((pure_images.shape))
    
    len_of_data = len(pure_images)//batch_size
    
    positionlist = []
    for i in range(len_of_data):
        pure_slice = pure_images[i*batch_size:(i+1)*batch_size]
        
        lr_images[i*batch_size:(i+1)*batch_size], indices = obscure_left_to_right(pure_slice, percent)
        
        positionlist.append(indices)
      
    return lr_images, positionlist

def gen_tb_data(pure_images, percent, batch_size):
    
    tb_images = np.zeros((pure_images.shape))
    
    len_of_data = len(pure_images)//batch_size
    for i in range(len_of_data):
        pure_slice = pure_images[i*batch_size:(i+1)*batch_size]
        
        tb_images[i*batch_size:(i+1)*batch_size] = obscure_top_to_bottom(pure_slice, percent)
      
    return tb_images

def gen_block_data(pure_images, num_blocks, blockdim, batch_size):
    
    block_images = np.zeros((pure_images.shape))
    
    len_of_data = len(pure_images)//batch_size
    
    positionlist = []
    for i in range(len_of_data):
        pure_slice = pure_images[i*batch_size:(i+1)*batch_size]
        
        block_images[i*batch_size:(i+1)*batch_size], positions = obscure_block(pure_slice, blockdim, num_blocks)
        
        positionlist.append(positions)
      
    return block_images, positionlist

def gen_corrupted_data(pure_images, amount_to_corrupt, batch_size):
  corrupted_images = np.zeros((pure_images.shape))
  len_of_data = len(pure_images)//batch_size
  for i in range(len_of_data):
    pure_slice = pure_images[i*batch_size:(i+1)*batch_size]
    
    corruption_list = random.sample(range(8), amount_to_corrupt)
                
    corrupted_images[i*batch_size:(i+1)*batch_size] = corrupt_images(pure_slice, corruption_list)
  
  return corrupted_images

def gen_photographer_dataset_range(pure_images, pure_labels, visual_analyzer, batch_size, amount_to_corrupt):
    
  restore_loss_target = np.zeros((len(pure_images), 8))
  
  corruption = gen_corrupted_data(pure_images, amount_to_corrupt, batch_size)
  
  for i in range(8):
      
      print(i)
      
      restoration = purify_images(corruption, pure_images, [i])
      
      predictions = visual_analyzer.predict(restoration)
            
      restore_loss_target[:, i] = tf.keras.losses.binary_crossentropy(pure_labels, predictions).numpy().reshape((50000))
      
  restore_loss_rounded = np.zeros((restore_loss_target.shape))
  
  for i in range(len(restore_loss_target)):
      
      restore_loss_rounded[i, np.argmin(restore_loss_target[i])] = 1
              
  return corruption, restore_loss_rounded

def gen_photographer_dataset_block(pure_images, pure_labels, visual_analyzer, batch_size, numblocks, blockdim):
    
  restore_loss_target = np.zeros((len(pure_images), numblocks))
  
  blockhead, positionlist = gen_block_data(pure_images, numblocks, blockdim, batch_size)
  
  batch_size = int(batch_size)
    
  for i in range(numblocks):
      
      print(i)
      
      for j in range(len(positionlist)):
                    
          restoration = restore_block(pure_images[j*batch_size:(j+1)*batch_size], blockhead[j*batch_size:(j+1)*batch_size], positionlist[j][0, i], positionlist[j][1, i], blockdim)
          
          predictions = visual_analyzer.predict(restoration)
                
          restore_loss_target[j*batch_size:(j+1)*batch_size, i] = tf.keras.losses.binary_crossentropy(pure_labels[j*batch_size:(j+1)*batch_size], predictions).numpy().reshape((batch_size))
  
  restore_loss_rounded = np.zeros((restore_loss_target.shape))
    
  for i in range(len(restore_loss_target)):
      
      restore_loss_rounded[i, np.argmin(restore_loss_target[i])] = 1

  return blockhead, restore_loss_rounded

def gen_photographer_dataset_sides(pure_images, pure_labels, visual_analyzer, batch_size, percent, percentrestore):
    
  restoreamount = int(32*percentrestore)
    
  restore_loss_target = np.zeros((len(pure_images), restoreamount+1))
  
  sides, positionlist = gen_lr_data(pure_images, percent, batch_size)
  
  batch_size = int(batch_size)
    
  for i in range(restoreamount+1):
      
      print(i)
      
      for j in range(len(positionlist)):
                    
          restoration = restore_left_to_right(pure_images[j*batch_size:(j+1)*batch_size], sides[j*batch_size:(j+1)*batch_size], positionlist[j][0], positionlist[j][1], i, restoreamount-i)
          
          predictions = visual_analyzer.predict(restoration)
                
          restore_loss_target[j*batch_size:(j+1)*batch_size, i] = tf.keras.losses.binary_crossentropy(pure_labels[j*batch_size:(j+1)*batch_size], predictions).numpy().reshape((batch_size))
  
  restore_loss_rounded = np.zeros((restore_loss_target.shape))
    
  for i in range(len(restore_loss_target)):
      
      restore_loss_rounded[i, np.argmin(restore_loss_target[i])] = 1
      
  return sides, restore_loss_rounded

























