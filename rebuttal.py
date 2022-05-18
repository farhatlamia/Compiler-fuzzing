import os 

def cleanDir(dir_name):
    sum_loc = 0 
    pp_, non_pp = [], []
    for root_, dirs, files_ in os.walk(dir_name):
       for file_ in files_:
           full_p_file = os.path.join(root_, file_)
           if(os.path.exists(full_p_file)):
             if (full_p_file.endswith('.jl')):
               pp_.append(full_p_file)
               sum_loc = sum_loc + len(open( full_p_file ).readlines(  ))

             else:
               non_pp.append(full_p_file)
    for f_ in non_pp:
        os.remove(f_)
    print( "="*50 )
    print( dir_name )
    print( 'removed {} non-Julia files, kept {} Julia files #savespace '.format(len(non_pp), len(pp_)) )
    print( "="*50 )
    print('Total LOC:' , sum_loc )
    print( "="*50 )

'''
bug count: 
arr = [6, 26, 2, 3, 4, 6, 5, 10, 33, 3, 5, 23, 2, 8, 85, 14, 8, 11, 13, 49] 
'''

'''

watch_count = [123, 12, 11, 6, 18, 11, 15, 6, 5, 12, 16, 11, 10, 5, 7, 52, 31, 10, 25]
star_count  = [2300, 36, 96, 19, 290, 36, 162, 52, 13, 71, 53, 184, 74, 15, 22, 243, 138, 22, 116] 
fork_count  = [375, 34, 28, 18, 43, 27, 39, 82, 9, 39, 11, 57, 17, 10, 9, 29, 48, 7, 55] 
commits     = [269, 252, 223, 327, 186, 658, 257, 250, 149, 204, 303, 727, 227, 644, 570, 415, 277, 591, 272, 223]
'''


if __name__ == '__main__':
        dir = '/Users/arahman/Documents/OneDriveWingUp/OneDrive-TennesseeTechUniversity/Research/SciSoft/BUG-CATEG/REVAMP/rebuttal-repos'
        cleanDir(dir)
