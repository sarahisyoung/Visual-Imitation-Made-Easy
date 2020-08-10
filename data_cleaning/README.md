# Data Processing Pipeline
> Option 1 is for converting raw video inputs directly into training data. Option 2 splits this process into separate steps.

## Option 1: Videos --> Labels
> To do step 3 in separate commands, see option 2.
1. Collect videos. 
2. Put videos into folder.
    This is what the `myVideos` folder structure should look like.
    ```
    .
    ├── myVideos                    
    │   ├── vids          
    │   │   ├── GH0101.mp4      
    │   │   ├── GH0102.mp4                
    │   │   
    │   └── ...     
    │   
    ```
    
3. Run script from `trash_bot` directory. This will split into frames, run colmap, and create colmap visualization videos.
    ```shell
    python data_cleaning/script_colmap.py --dir myVideos/ --write saveVizVideosHere --split 1
    ```
    
4. This is what the `myVideos` folder will look like after videos are split. 
    Colmap will then automatically run on the `frames` folder.
    ```
    .
    ├── myVideos                    
    │   ├── vids          
    │   │   ├── GH0101.mp4      
    │   │   ├── GH0102.mp4      
    │   │   ├── ...
    │   │          
    │   ├── frames          
    │   │   ├── GH0101_frames 
    │   │   │   ├── theframe001.png   
    │   │   │   ├── theframe002.png     
    │   │   │   ├── ...
    │   │   ├── GH0102_frames                
    │   │   
    │   └── ...     
    │   
    ```
5. Clean data!! 
    - Look through viz videos and delete bad images from the original `images folder`.
 
## Option 2: Videos --> Frames --> Extract Labels --> Visualize
1. Collect videos.
2. Split videos (or single video `--f 0` )into frames. Saves to `--save ` directory.
    ```shell
     cd trash_bot
     python data_cleaning/split_video.py --dir data/vids --f 1 --save video_frames
    ```

3. Run COLMAP to get ground truth labels. See below for details.
    ```shell
     DATASET_PATH=~/may6th/frames/GH061507_frames/
     
     colmap automatic_reconstructor \
    --workspace_path $DATASET_PATH \
    --image_path $DATASET_PATH/images
    ```
4. Convert COLMAP `bin` outputs to `txt` files.
    ```shell
    colmap model_converter --input_path ~/may6th/frames/GH061507_frames/sparse/0/ --output_path ~/may6th/frames/GH061507_frames/sparse/0/ --output_type TXT
    ```
5. Write relative translations and rotations to labels.json.
    ```shell
    python data_cleaning/write_translations.py --dir ~/may6th/frames/GH061507_frames/
    ```
    
6. Visualize translations and clean ground truth labels.  
    ```shell
    python data_cleaning/labels_tester.py --dir ~/may6th/frames/GH061507_frames/ --write ~/may_videos/colmap_results/
    ```  
 


### Running Colmap:
> For getting ground truth labels.

1. Create a folder of runs. 
    ```
    .
    ├── frames                    
    │   ├── GH0001          # Run 1 folder
    │   │   ├── images      # images folder          
    │   │   
    │   ├── GH0002          # Run 2 folder
    │   │   ├── images      # images folder          
    │   │   
    │   └── ...     
    │   
    ```
2. Run script. This runs COLMAP, writes `labels.json` file, 
        outputs video for COLMAP visualization, and removes `dense/stereo` folder. 
        Provide output folder in ```--write``` for colmap visualization videos.
    ```shell
     python scripts/script_colmap.py --dir ~/may6th/frames/ \
           --write ~/may6th/colmap_done_videos/ --split 0
    ```

3. Your directory should look like this.
    ```
    .
    ├── frames                    
    │   ├── GH0001              # Run 1 folder
    │   │   ├── images          # images folder  
    │   │   ├── labels.json     # labels file    
    │   │   ├── sparse          # COLMAP output        
    │   │   ├── ...          
    │   │   
    │   ├── GH0002              # Run 2 folder
    │   │   ├── images          # images folder    
    │   │   ├── labels.json     # labels file    
    │   │   ├── sparse          # COLMAP output        
    │   │   ├── ...          
    │   │   
    │   └── ...     
    │   
    ```


#### To use masking with COLMAP:
> For masking out the gripper for better COLMAP results.

1. Create mask. Provide directory with all runs that need masking. 
    ```shell
     python data_cleaning/mask.py --dir ~/may6th/finished/usemask/
    ```
    This saves a folder called `mask_images` inside each run folder. 
    Double check the mask is in the correct spot.

2. Run COLMAP. See COLMAP section for more details.
    ```shell
     python scripts/script_colmap.py --dir ~/may6th/finished/usemask/ --write ~/may6th/colmap_done_videos/ --use_mask 1
    ```
    
