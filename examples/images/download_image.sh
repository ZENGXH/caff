#!/usr/bin/env sh
# down load several image

CHILD=http://farm3.static.flickr.com/2240/1504343937_fc94985d78.jpg
FLOWER=http://farm3.static.flickr.com/2804/4319240757_6c54ed0d72.jpg
DOG=http://farm4.static.flickr.com/3246/2743075753_d291f0f4dc.jpg

echo "down image to $CHILD examples/images..."

wget $CHILD -O child.jpg
wget $FLOWER -O flower.jpg
wget $DOG -O dog.jpg


echo "done"

