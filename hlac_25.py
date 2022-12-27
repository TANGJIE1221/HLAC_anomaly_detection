from numba import guvectorize

#calculate 25 hlac features 
#use guvectorize to speed up


@guvectorize(['void(u1[:,:], u8[:], u8[:])'], '(x,y),(z)->(z)', nopython=True)
def hlac_25(img, a,feature):
  tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9 = 0,0,0,0,0,0,0,0,0,0
  tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tmp16, tmp17, tmp18, tmp19 = 0,0,0,0,0,0,0,0,0,0
  tmp20, tmp21, tmp22, tmp23, tmp24 = 0,0,0,0,0
  

  for y in range(img.shape[0]-2):
    for x in range(img.shape[1]-2):
      # 0th order
      tmp0 += img[y+1,x+1]
      # 1st order
     
      tmp1 += img[y+1,x+1] * img[y+1, x+2] # -..
      tmp2 += img[y+1,x+1] * img[y, x+2]   # -./
      tmp3 += img[y+1,x+1] * img[y, x+1]   # -.|
      tmp4 += img[y+1,x+1] * img[y, x]     # \.-
      # 2nd order
      tmp5 += img[y+1,x+1] * img[y+1, x+2] * img[y+1, x]
      tmp6 += img[y+1,x+1] * img[y, x+2] * img[y+2, x]
      tmp7 += img[y+1,x+1] * img[y, x+1]  * img[y+2, x+1]
      tmp8 += img[y+1,x+1] * img[y, x] * img[y+2, x+2]
      tmp9 += img[y+1,x+1] * img[y, x+2] * img[y+1, x]
      
      tmp10 += img[y+1,x+1] * img[y, x+1]  * img[y+2, x]
      tmp11 += img[y+1,x+1] * img[y, x] * img[y+2, x+1]
      tmp12 += img[y+1,x+1] * img[y+2, x+2] * img[y+1, x]
      tmp13 += img[y+1,x+1] * img[y+1, x+2] * img[y+2, x]
      tmp14 += img[y+1,x+1] * img[y, x+2] * img[y+2, x+1]

      tmp15 += img[y+1,x+1] * img[y, x+1]  * img[y+2, x+2]
      tmp16 += img[y+1,x+1] * img[y+1, x+2] * img[y, x]
      tmp17 += img[y+1,x+1] * img[y, x+1]  * img[y+1, x]
      tmp18 += img[y+1,x+1] * img[y, x] * img[y+2, x]
      tmp19 += img[y+1,x+1] * img[y+1, x] * img[y+2, x+1]

      tmp20 += img[y+1,x+1] * img[y+2, x+2] * img[y+2, x]
      tmp21 += img[y+1,x+1] * img[y+1, x+2] * img[y+2, x+1]
      tmp22 += img[y+1,x+1] * img[y, x+2] * img[y+2, x+2]
      tmp23 += img[y+1,x+1] * img[y, x+1]  * img[y+1, x+2]
      tmp24 += img[y+1,x+1] * img[y, x+2] * img[y, x]
  feature[0] = tmp0
  feature[1] = tmp1
  feature[2] = tmp2
  feature[3] = tmp3
  feature[4] = tmp4
  feature[5] = tmp5
  feature[6] = tmp6
  feature[7] = tmp7
  feature[8] = tmp8
  feature[9] = tmp9
  feature[10] = tmp10
  feature[11] = tmp11
  feature[12] = tmp12
  feature[13] = tmp13
  feature[14] = tmp14
  feature[15] = tmp15
  feature[16] = tmp16
  feature[17] = tmp17
  feature[18] = tmp18
  feature[19] = tmp19
  feature[20] = tmp20
  feature[21] = tmp21
  feature[22] = tmp22
  feature[23] = tmp23
  feature[24] = tmp24