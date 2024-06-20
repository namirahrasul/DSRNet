import dominate #used for creating and manipulating HTML documents in a programmatic way.
from dominate.tags import * #easier to create HTML tags directly.
import os

#ALL reflesh should be refresh
class HTML:
    def __init__(self, web_dir, title, reflesh=0): #reflesh: refresh_rate
        self.title = title #title of web page
        self.web_dir = web_dir#parent directory of all web content e.g images
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        #(mingcv) print(self.img_dir)

        self.doc = dominate.document(title=title)
        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(reflesh)) #was "reflesh" in mingcv
        #The http-equiv attribute provides an HTTP header for the information/value of the content attribute.In <head></head>
    def get_image_dir(self):
        return self.img_dir

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txts, links, height=400):# add images to the HTML document within a table. Each cell (td) in the table contains an image on one line and its descriptive text on the next line.
        self.add_table()
        with self.t:#opens the table context for adding rows
            with tr():#open table row context
                for im, txt, link in zip(ims, txts, links):#This iterates over the images, texts, and links
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):
                                img(style="height:%dpx" % height, src=os.path.join('images', im))
                            br()
                            p(txt)

    def save(self):# save the generated HTML document to a file.
        html_file = '%s/index.html' % self.web_dir  # self.web_dir value will be placed in %s like sprintf(html_file, “%s/index.html”, web_dir);
        f = open(html_file, 'wt')#write text mode 
        f.write(self.doc.render())# writes the rendered HTML content to the file. The render method of the doc object (which is an instance of the dominate.document class) converts the HTML document into a string.
        f.close()


if __name__ == '__main__':#if the script is being run as the main module. This block will only execute if the script is run directly (not imported as a module).

    html = HTML('web/', 'test_html')#'web/'=web_dir, 'test_html'=title
    html.add_header('hello world')

    ims = []#ims for image filenames, txts for image descriptions, and links for image links.
    txts = []
    links = []
    for n in range(4):#adds images, texts and links to the list so that they can be added to html. 
        '''??? why 4'''
        ims.append('image_%d.png' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.png' % n)
    html.add_images(ims, txts, links)
    html.save()
