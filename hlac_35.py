from numba import guvectorize

#calculate 35 hlac features
#use guvectorize to speed up

@guvectorize(['void(u1[:,:], u8[:], u8[:])'], '(x,y),(z)->(z)', nopython=True)
def hlac_35(img, a,feature):
  tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9 = 0,0,0,0,0,0,0,0,0,0
  tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tmp16, tmp17, tmp18, tmp19 = 0,0,0,0,0,0,0,0,0,0
  tmp20, tmp21, tmp22, tmp23, tmp24, tmp25, tmp26, tmp27, tmp28, tmp29 = 0,0,0,0,0,0,0,0,0,0
  tmp30, tmp31, tmp32, tmp33, tmp34 = 0,0,0,0,0

  for y in range(img.shape[0]-2):
    for x in range(img.shape[1]-2):
      # 0th order
      tmp0 += img[y+1,x+1]
      # 1st order
      tmp1 += img[y+1,x+1] * img[y+1,x+1]
      tmp2 += img[y+1,x+1] * img[y+1, x+2] # -..
      tmp3 += img[y+1,x+1] * img[y, x+2]   # -./
      tmp4 += img[y+1,x+1] * img[y, x+1]   # -.|
      tmp5 += img[y+1,x+1] * img[y, x]     # \.-
      # 2nd order
      tmp6 += img[y+1,x+1] * img[y+1,x+1] * img[y+1,x+1]
      tmp7 += img[y+1,x+1] * img[y+1, x+2] * img[y+1,x+1]
      tmp8 += img[y+1,x+1] * img[y+1, x+2] * img[y+1, x+2]
      tmp9 += img[y+1,x+1] * img[y, x+2] * img[y+1,x+1]
      tmp10 += img[y+1,x+1] * img[y, x+2] * img[y, x+2]
      tmp11 += img[y+1,x+1] * img[y, x+1]  * img[y+1,x+1]
      tmp12 += img[y+1,x+1] * img[y, x+1]  * img[y, x+1]
      tmp13 += img[y+1,x+1] * img[y, x] * img[y+1,x+1]
      tmp14 += img[y+1,x+1] * img[y, x] * img[y, x]

      tmp15 += img[y+1,x+1] * img[y+1, x+2] * img[y+1, x]
      tmp16 += img[y+1,x+1] * img[y, x+2] * img[y+2, x]
      tmp17 += img[y+1,x+1] * img[y, x+1]  * img[y+2, x+1]
      tmp18 += img[y+1,x+1] * img[y, x] * img[y+2, x+2]
      tmp19 += img[y+1,x+1] * img[y, x+2] * img[y+1, x]
      
      tmp20 += img[y+1,x+1] * img[y, x+1]  * img[y+2, x]
      tmp21 += img[y+1,x+1] * img[y, x] * img[y+2, x+1]
      tmp22 += img[y+1,x+1] * img[y+2, x+2] * img[y+1, x]
      tmp23 += img[y+1,x+1] * img[y+1, x+2] * img[y+2, x]
      tmp24 += img[y+1,x+1] * img[y, x+2] * img[y+2, x+1]

      tmp25 += img[y+1,x+1] * img[y, x+1]  * img[y+2, x+2]
      tmp26 += img[y+1,x+1] * img[y+1, x+2] * img[y, x]
      tmp27 += img[y+1,x+1] * img[y, x+1]  * img[y+1, x]
      tmp28 += img[y+1,x+1] * img[y, x] * img[y+2, x]
      tmp29 += img[y+1,x+1] * img[y+1, x] * img[y+2, x+1]

      tmp30 += img[y+1,x+1] * img[y+2, x+2] * img[y+2, x]
      tmp31 += img[y+1,x+1] * img[y+1, x+2] * img[y+2, x+1]
      tmp32 += img[y+1,x+1] * img[y, x+2] * img[y+2, x+2]
      tmp33 += img[y+1,x+1] * img[y, x+1]  * img[y+1, x+2]
      tmp34 += img[y+1,x+1] * img[y, x+2] * img[y, x]
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
  feature[25] = tmp25
  feature[26] = tmp26
  feature[27] = tmp27
  feature[28] = tmp28
  feature[29] = tmp29
  feature[30] = tmp30
  feature[31] = tmp31
  feature[32] = tmp32
  feature[33] = tmp33
  feature[34] = tmp34