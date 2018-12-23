You can access HDFS using philly-fs. This executable is ONLY for Linux machines. You don't need to install Python or anything to run it.

We strongly RECOMMEND you mount this file share to your Linux box instead of copying the files to your machine. The biggest benefit to this is if we make any bug fixes, you will get them for free.
1)  Install cifs utils if you don't have it:
    sudo apt-get install cifs-utils
    
2)  Create a directory on your local machine where you want to mount the Windows File Share to:
    sudo mkdir /mnt/philly-fs
    sudo chmod 755 /mnt/philly-fs
    
3)  Mount the Windows File Share on your local machine (replace <your alias> to your Microsoft alias):
    sudo mount -t cifs -o username=<your alias>,domain=redmond,file_mode=0777,dir_mode=0777 //scratch2.ntdev.corp.microsoft.com/scratch/Philly/philly-fs /mnt/philly-fs

    sudo mount -t cifs -o username=yuyua,domain=redmond,file_mode=0777,dir_mode=0777 //scratch2.ntdev.corp.microsoft.com/scratch/Philly/philly-fs /mnt/philly-fs    
    
    If you see "mount error(112):Host is down" when performing the above command, please try mounting using the following command:
    sudo mount -t cifs -o username=<your alias>,domain=redmond,file_mode=0777,dir_mode=0777,vers=2.0 //scratch2.ntdev.corp.microsoft.com/scratch/Philly/philly-fs /mnt/philly-fs
    
    If the above one still fails, please replace vers=2.0 to vers=3.0, so run the following command:
    sudo mount -t cifs -o username=<your alias>,domain=redmond,file_mode=0777,dir_mode=0777,vers=3.0 //scratch2.ntdev.corp.microsoft.com/scratch/Philly/philly-fs /mnt/philly-fs
    
    If the mount still fails, please contact the team at the email address listed below.
    
4)  For your convenience, create the following alias (bash) and add it to your ~/.bashrc: (See section below on "Python versions of philly-fs.pyc and where you can find them" when creating this alias)
    alias philly-fs='python /mnt/linux/philly-fs'

Running philly-fs:
Once you have mounted the above file share (and assuming you created the alias suggested above), here is how your run philly-fs.

1) Using bash script which looks for latest philly-fs and runs the latest executable (We highly recommend this option)
   bash philly-fs.bash -h
   bash philly-fs.bash -ls //philly/rr1/pnrsy 

2) You can also use the direct executable. But it is NOT recommended. 
   philly-fs -h
   philly-fs -ls //philly/rr1/pnrsy 
===========================================================
Help? Feedback?

If there are any issues, ping phillyinfra@microsoft.com
