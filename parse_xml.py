# coding=utf-8
import datetime
import xml.etree.ElementTree as ET

if __name__ == '__main__':
    filename = "part_2"
    root = ET.parse('./data/%s.xml' % filename).getroot()
    keys_by_type = {}
    data_by_type = {}

    for xmldicts in root.findall("array/dict"):
        attrs = iter(xmldicts.findall("*"))
        xmldict = {}
        for e in attrs:
            if e.tag == 'key':
                val = attrs.next()
                if val.tag == 'real':
                    xmldict[e.text] = val.text
                elif val.tag == 'string':
                    xmldict[e.text] = val.text.encode("utf-8")
                elif val.tag == 'date':
                    xmldict[e.text] = datetime.datetime\
                        .strptime(val.text, '%Y-%m-%dT%H:%M:%SZ')\
                        .strftime("%Y-%m-%d %H:%M:%S")
                #int(datetime.datetime.strptime(val.text, '%Y-%m-%dT%H:%M:%SZ').strftime("%s"))

        if xmldict['type'] not in keys_by_type:
            keys_by_type[xmldict['type']] = []
        if xmldict['type'] not in data_by_type:
            data_by_type[xmldict['type']] = []

        for (k,v) in xmldict.iteritems():
            if k != 'type':
                if k not in keys_by_type[xmldict['type']]:
                    keys_by_type[xmldict['type']].append(k)
        data_by_type[xmldict['type']].append(xmldict)

    for (type, keys) in keys_by_type.iteritems():
        if type in ('debug', 'debugSlide', 'deviceInfo'):
            continue
        with open('./data/csvs/%s_%s.csv' % (filename, type),'w') as f:
            f.write(','.join(keys))
            f.write("\n")
            for x in data_by_type[type]:
                r = []
                for k in keys:
                    r.append(x[k])
                f.write(','.join(r))
                f.write("\n")



