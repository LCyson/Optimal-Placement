# Optimal Placement

Optimal Placement is a Python project that attempts a vary of methods to solve the
stock optimal placement problem. 

## Local setup
- Pull this repository from github, and open it in a Python Compatible IDE (Pycharm, Virtual Studio, or etc)
- Install any missing dependencies/modules in the IDE. You should be hinted to do it in the IDE.

## Usage
- Run message_parser.py to download and process itch data 
    - Note: it may take up to two hours for the download/cleaning to be done. You can also download the 
    07302019.NASDAQ_ITCH50.zip file from ftp://emi.nasdaq.com/ITCH/ and put it under the /data and let the 
    program unzip it.
- Run build-lob.py to dynamically create a LOB, visual part for this is TBD
    - Note: you may need enough disk space on your laptop to do this.

## Troubleshooting
1. For any "command not found" or "module not found" error, please follow the instruction
in the error message to install the corresponding module/package. More details can be found
on Google.  

## License
[MIT](https://choosealicense.com/licenses/mit/)