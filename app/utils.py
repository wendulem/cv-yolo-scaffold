def allowed_file(filename, ALLOWED_EXTENSIONS=set(['png', 'jpg', 'jpeg'])):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# function to add correct and or comma
def and_syntax(alist):
    if len(alist) == 1:
        alist = "".join(alist)
        return alist
    elif len(alist) == 2:
        alist = " and ".join(alist)
        return alist
    elif len(alist) > 2:
        alist[-1] = "and " + alist[-1]
        alist = ", ".join(alist)
        return alist
    else:
        return
