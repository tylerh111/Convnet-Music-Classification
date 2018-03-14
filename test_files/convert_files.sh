#!/bin/bash

# example
# $ /usr/bin/sox "../audiofiles/01 - Symphonic Variations.flac" -n spectrogram -o "symphonic_variations.png"

AUDIO_PATH="audiofiles"
SPECT_PATH="input"
CLEAN=false

# get command line args/options
POSITIONAL=()
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        -c|--clean)
            CLEAN=true
            shift # past argument
            shift # past value
            ;;
    esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters


# for each class name
for dir in $(ls "$AUDIO_PATH/") ;
do
    printf "Directory: $dir\n"

    # clean the directory
    # removes spaces
    if $CLEAN ; then
        printf "Cleaning directory\n"
        find -name "* *" -type f | rename 's/ /_/g'
    fi


    # convert flac -> png (spectrogram)
    for fn in $(ls "$AUDIO_PATH/$dir/") ;
    do
        printf "\tCreating spectrogram of $fn\n"
        no_ext=${fn%.flac} #drop extension for png
        /usr/bin/sox "$AUDIO_PATH/$dir/$fn" -n spectrogram -r -a -o "$SPECT_PATH/$dir/$no_ext.png"
    done;


done;






