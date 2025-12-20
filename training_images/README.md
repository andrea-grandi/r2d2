# Training Images for TrackerML

Place images of the object you want to track in this folder.

## Supported formats:
- .jpg, .jpeg
- .png
- .bmp
- .tiff

## Tips for good training images:
1. Take 5-15 images of the object from different angles
2. Use different lighting conditions
3. Include some background variety
4. Make sure the object is clearly visible
5. Use images with resolution similar to your camera

## Example:
```
training_images/
  ├── object_front.jpg
  ├── object_side.jpg
  ├── object_top.jpg
  ├── object_angle1.jpg
  └── object_angle2.jpg
```

The tracker will automatically load all images from this folder when started.
