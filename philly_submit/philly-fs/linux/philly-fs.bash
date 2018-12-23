#!/bin/bash

VersionApi="https://philly/api/philly-fs/version?client=linux&preview="
DownloadApi="https://philly/api/philly-fs/download?client=linux&version=&preview="
phillyFsRootFolderPath="/tmp/philly-fs"
phillyFsTimestampPath="$phillyFsRootFolderPath/timestamp.txt"
PhillyFsVersionRegEx="^[0-9]+\\.[0-9]+\\.[0-9]+"
PhillyFsFileName="philly-fs"

PhillyFsLogPrefix="philly-fs.bash:"
ShowVerbose="$PHILLY_VERBOSE"

verbose() { 
    if (( ShowVerbose == 1 )); then
        echo "$PhillyFsLogPrefix $(date +%H:%M:%S.%N): V: $*"
    fi
}

warning() {
    if (( ShowVerbose == 1 )); then
        echo "$PhillyFsLogPrefix $(date +%H:%M:%S.%N): W: $*"
    fi
}

error() {
    echo "$PhillyFsLogPrefix $(date +%H:%M:%S.%N): E: $*" 1>&2;
}

getPhillyFsRootPath() {
    if [[ $# -lt 1 ]]; then 
        error "please pass at least one parameter"
        return 1
    fi

    local -n rootPath=$1
    if ((isPreview == 1)); then
        rootPath="/tmp/philly-fs-preview"
    else
        rootPath="/tmp/philly-fs"
    fi
}

getPhillyFsTimestampPath() {
    if [[ $# -lt 1 ]]; then 
        error "please pass at least one parameter"
        return 1
    fi

    local -n timestampPath=$1
    local rootPath=
    getPhillyFsRootPath rootPath
    timestampPath="$rootPath/timestamp.txt"
}

getPhillyFsVersionPath() {
    if [[ $# -lt 1 ]]; then 
        error "please pass at least one parameter"
        return 1
    fi

    local -n versionPath=$1
    local rootPath=
    getPhillyFsRootPath rootPath
    versionPath="$rootPath/version.txt"
}

updateTimeStamp() {
    currTimeStamp=$(date +%s)

    timestampPath=
    getPhillyFsTimestampPath timestampPath

    verbose "Updating $timestampPath..."
    echo "$currTimeStamp" > $timestampPath
}

getLocalTimeStamp() {
    if [[ $# -lt 1 ]]; then 
        error "please pass at least one parameter"
        return 1
    fi

    local -n versionTimestamp=$1

    timestampPath=
    getPhillyFsTimestampPath timestampPath

    if [[ -f "$timestampPath" ]]; then
        read -r versionTimestamp < "$timestampPath"
    fi
}

getLocalVersion() {
    if [[ $# -lt 1 ]]; then 
        error "please pass at least one parameter"
        return 1
    fi

    local -n version=$1
    verPath=
    getPhillyFsVersionPath verPath

    if [[ -f "$verPath" ]]; then
        read -r version < "$verPath"
    fi
}

updateVersion() {
    if [[ $# -lt 1 ]]; then 
        error "please pass at least one parameter"
        return 1
    fi

    verPath=
    getPhillyFsVersionPath verPath

    verbose "Updating $verPath..."
    echo $1 > "$verPath"
}

installPhillyFs() {
    # validating sufficient arguments were passed
    if [[ $# -lt 2 ]]; then
        warning "Expecting 2 arguments to installPhillyFs but got less than that!"
        return 1
    fi

    version=$1
    updateMetaData=$2
    if [[ ! "$version" =~ $PhillyFsVersionRegEx ]]; then
        warning "First argument should be version to install!"
        return 1
    fi

    if (( updateMetaData != 1 && updateMetaData != 0 )); then
        warning "Second argument must have value of 0 or 1"
        return 1
    fi

    # creating $phillyFsRootFolderPath if it doesn't exists
    if [ ! -d "$phillyFsRootFolderPath" ]; then
        verbose "Creating local philly-fs root folder..."
        mkdir "$phillyFsRootFolderPath"
        if [ ! -d "$phillyFsRootFolderPath" ]; then
            error "Unable to create $phillyFsRootFolderPath. Please check you have permissions to write to /tmp!"
            return 1
        fi
    fi

    # creating folder for philly-fs
    phillyFsFolderPath="$phillyFsRootFolderPath/$version"
    if [ -d "$phillyFsFolderPath" ]; then
        warning "$phillyFsFolderPath already exists. Deleting it..."
        rm -rf "$phillyFsFolderPath"
    fi
    verbose "Creating $phillyFsFolderPath for storing version $version of philly-fs!"
    mkdir "$phillyFsFolderPath"
    if [ ! -d "$phillyFsFolderPath" ]; then
        error "Unable to create $phillyFsFolderPath. Please check you have permissions to write to $phillyFsRootFolderPath!"
        return 1
    fi

    # downloading philly-fs.exe to folder
    phillyFsPath="$phillyFsFolderPath/$PhillyFsFileName"
    if ((isPreview == 1)); then
        api="${DownloadApi/version=&preview=/version=$version&preview=true}"
    else
        api="${DownloadApi/version=&preview=/version=$version&preview=false}"
    fi

    echo "$PhillyFsLogPrefix $(date +%H:%M:%S.%N): I: Downloading $version of $PhillyFsFileName from $api to $phillyFsPath..."
    echo "$PhillyFsLogPrefix $(date +%H:%M:%S.%N): I: curl -k \"$api\" -o \"$phillyFsPath\" --retry 5"
    curl -k "$api" -o "$phillyFsPath" --retry 5

    # verifying download was successful
    if [ -e "$phillyFsPath" ]; then
        # checking the file size to be at least 1 MB otherwise it is not philly-fs executable 
        # but instead some error message encountered while downloading philly-fs
        phillyFsSize=$(stat -c %s "$phillyFsPath")

        # checking if phillyFsSize is an integer
        if [[ $phillyFsSize =~ ^[0-9]+$ ]]; then
            # checking file is at least 1 MB
            if (( phillyFsSize > 1024*1024 )); then
                echo "$PhillyFsLogPrefix $(date +%H:%M:%S.%N): I: $PhillyFsFileName was successfully downloaded to $phillyFsPath!"

                # enabling execute permissions on philly-fs.exe
                verbose "Setting execute permission on $PhillyFsFileName..."
                chmod u+x "$phillyFsPath"

                # updating version.txt and timestamp.txt if updateMetadata is set
                if (( updateMetaData == 1 )); then
                    updateVersion "$latestVersion"
                    updateTimeStamp
                fi

                return 0
            else
                # delete this folder as it may cause issue in the future on whether this version of
                # philly-fs was downloaded or not
                error "$PhillyFsFileName failed to download! Error: Invalid file size!"
                rm -rf "$phillyFsFolderPath"
                return 1
            fi
        else
            error "$PhillyFsFileName failed to download! Error: Unable to determine file size!"
        fi
    else
        error "$PhillyFsFileName failed to download! Error: No file found at $phillyFsPath!"
        return 1
    fi
}

phillyFsPath=''
phillyFsArgs=''
localVersion=''
isUserVersion=0
isExpired=1
isPreview=0

# parsing arguments
while [[ $# -gt 0 ]]; do
    if [ "$1" == "-p" ] || [ "$1" == "--preview" ] ; then
        isPreview=1
        verbose "currently is running preview version of philly-fs "
    elif [ "$1" == "-z" ] || [ "$1" == "--usezip" ] ; then
        warning "Linux version doesn't support the option of '-z', will skip it"
    else
        phillyFsArgs="$phillyFsArgs $1"
    fi
    shift
done

getPhillyFsRootPath phillyFsRootFolderPath
if [[ "$isUserVersion" -eq 0 ]]; then
    getLocalVersion localVersion

    # testing localVersion matches regex
    if [[ "$localVersion" =~ $PhillyFsVersionRegEx ]]; then
        verbose "Local version: $localVersion"
        # generating and validating philly-fs path
        phillyFsPath="$phillyFsRootFolderPath/$localVersion/$PhillyFsFileName"
        if [ ! -f "$phillyFsPath" ]; then
            phillyFsPath=''
        fi
    else
        localVersion=''
    fi

    getPhillyFsTimestampPath phillyFsTimestampPath
    # determining amount of time since last version check. Calculate this value
    # only if there was a philly-fs found on local disk. Otherwise, don't bother.
    if [[ -f "$phillyFsPath" && -f "$phillyFsTimestampPath" ]]; then
        verbose "Fetching local version timestamp..."
        versionTimestamp=''
        getLocalTimeStamp versionTimestamp
    
        if [[ -n "$versionTimestamp" ]]; then
            currentTimestamp=$(date +%s)
            verbose "Comparing last checked timestamp: $versionTimestamp against current timestamp: $currentTimestamp"

            timestampDiff=$((currentTimestamp - versionTimestamp))
            verbose "Number of seconds since last checked: $timestampDiff"

            # check for new version every 24 hours
            if (( timestampDiff > 24*60*60 )); then
                isExpired=1
            else
                isExpired=0
            fi
        fi
    
        verbose "Timestamp Expired? $isExpired"
    fi

    if [[ -z "$phillyFsPath" || "$isExpired" == 1 ]]; then
        # clearing out previous value of phillyFsPath so it does not impact
        # the upcoming checks
        phillyFsPath=''

        verbose "Fetching remote version..."
        if ((isPreview == 1)); then
            latestVersion=$(curl -k "${VersionApi/preview=/preview=true}" --retry 5 2> /dev/null)
        else
            latestVersion=$(curl -k "${VersionApi/preview=/preview=false}" --retry 5 2> /dev/null)
        fi
        verbose "Latest version is $latestVersion"

        # checking the version meets the expected format. This is done to verify
        # that we got a valid response from API
        if [[ "$latestVersion" =~ $PhillyFsVersionRegEx ]]; then
            if [[ -n "$localVersion" && "$latestVersion" == "$localVersion" ]]; then
                phillyFsPath="$phillyFsRootFolderPath/$localVersion/$PhillyFsFileName"

                # checking if philly-fs exists or not
                if [[ -f "$phillyFsPath" ]]; then
                    verbose "Local version is latest already!"
                    updateTimeStamp
                else
                    # versions match however the binary is not found, so clear current value of
                    # phillyFsPath to force download fetched version
                    phillyFsPath=''
                fi
            else
                # local version does not match fetched version, so clear current value of 
                # phillyFsPath to force download fetched version
                phillyFsPath=''
            fi

            if [[ -z "$phillyFsPath" ]]; then
                # checking exit code of function to see if installation was successful or not
                if installPhillyFs "$latestVersion" 1; then
                    phillyFsPath="$phillyFsRootFolderPath/$latestVersion/$PhillyFsFileName"
                else
                    phillyFsPath=''
                fi
            fi
        fi
        # if by this point phillyFsPath was not set, that means we tried to download but failed
        if [[ -z "$phillyFsPath" || ! -e "$phillyFsPath" ]]; then
            error "No $PhillyFsFileName was found on machine and downloading $PhillyFsFileName from https://philly failed!"
            exit 1
        fi
    else
        verbose "Using cached path."
    fi
fi

verbose "Invoking $phillyFsPath with $phillyFsArgs..."
"$phillyFsPath" $phillyFsArgs
