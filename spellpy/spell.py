import re
import os
import sys
import pickle
from threading import Timer
import pandas as pd
import hashlib
from datetime import datetime
import string
import logging

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s][%(levelname)s]: %(message)s')


sys.setrecursionlimit(10000)


class LCSObject:
    """ Class object to store a log group with the same template
    """
    def __init__(self, logTemplate='', logIDL=[]):
        self.logTemplate = logTemplate
        self.logIDL = logIDL


class Node:
    """ A node in prefix tree data structure
    """
    def __init__(self, token='', templateNo=0):
        self.logClust = None
        self.token = token
        self.templateNo = templateNo
        self.childD = dict()


class CustomUnpickler(pickle.Unpickler):
    """ CustomUnpickler is to prevent can't get attribute error when pickle load.
    """
    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)


class LogParser(pickle.Unpickler):
    """ LogParser class
    Attributes
    ----------
        path : the path of the input file
        logName : the file name of the input file
        savePath : the path of the output file
        tau : how much percentage of tokens matched to merge a log message.
            1: Merge as much as possible
            0: Differentiate as much as possible
    """
    def __init__(self, indir='./', outdir='./result/', log_format=None, tau=0.5, keep_para=True, text_max_length=4096, logmain=None):
        self.path = indir
        self.logname = None
        self.logmain = logmain
        self.savePath = outdir
        self.tau = tau
        self.logformat = log_format
        self.df_log = None
        self.keep_para = keep_para
        self.lastestLineId = 0
        self.text_max_length = text_max_length
        self.rootNode = None
        self.logCluL = None

        self.headers, self.regex = self.generate_logformat_regex(self.logformat)

        rootNodePath = os.path.join(self.savePath, 'rootNode.pkl')
        logCluLPath = os.path.join(self.savePath, 'logCluL.pkl')

        if os.path.exists(rootNodePath) and os.path.exists(logCluLPath):
            with open(rootNodePath, 'rb') as f:
                self.rootNode = CustomUnpickler(f).load()
            with open(logCluLPath, 'rb') as f:
                self.logCluL = CustomUnpickler(f).load()
            self.lastestLineId = 0
            self.setLatestLineId()
            logging.info(f'Load objects done, lastestLineId: {self.lastestLineId}')
        else:
            self.rootNode = Node()
            self.logCluL = []
            self.lastestLineId = 0

    def setDataframe(self, df):
        self.df_log = df

    def setLatestLineId(self):
        for logclust in self.logCluL:
            Max = max(logclust.logIDL)
            if Max > self.lastestLineId:
                self.lastestLineId = Max

    def LCS(self, seq1, seq2):
        lengths = [[0 for j in range(len(seq2)+1)] for i in range(len(seq1)+1)]
        # row 0 and column 0 are initialized to 0 already
        for i in range(len(seq1)):
            for j in range(len(seq2)):
                if seq1[i] == seq2[j]:
                    lengths[i+1][j+1] = lengths[i][j] + 1
                else:
                    lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])

        # read the substring out from the matrix
        result = []
        lenOfSeq1, lenOfSeq2 = len(seq1), len(seq2)
        while lenOfSeq1 != 0 and lenOfSeq2 != 0:
            if lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1-1][lenOfSeq2]:
                lenOfSeq1 -= 1
            elif lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1][lenOfSeq2-1]:
                lenOfSeq2 -= 1
            else:
                assert seq1[lenOfSeq1-1] == seq2[lenOfSeq2-1]
                result.insert(0, seq1[lenOfSeq1-1])
                lenOfSeq1 -= 1
                lenOfSeq2 -= 1
        return result

    def SimpleLoopMatch(self, logClustL, seq):
        for logClust in logClustL:
            if float(len(logClust.logTemplate)) < 0.5 * len(seq):
                continue
            # Check the template is a subsequence of seq (we use set checking as a proxy here for speedup since
            # incorrect-ordering bad cases rarely occur in logs)
            token_set = set(seq)
            if all(token in token_set or token == '<*>' for token in logClust.logTemplate):
                return logClust
        return None

    def PrefixTreeMatch(self, parentn, seq, idx):
        retLogClust = None
        length = len(seq)
        for i in range(idx, length):
            if seq[i] in parentn.childD:
                childn = parentn.childD[seq[i]]
                if (childn.logClust is not None):
                    constLM = [w for w in childn.logClust.logTemplate if w != '<*>']
                    if float(len(constLM)) >= self.tau * length:
                        return childn.logClust
                else:
                    return self.PrefixTreeMatch(childn, seq, i + 1)

        return retLogClust

    def LCSMatch(self, LCSMap, seq):
        retLCSObject = None

        maxLen = -1
        maxLCSObject = None
        set_seq = set(seq)
        size_seq = len(seq)
        for LCSObject in LCSMap:
            set_template = set(LCSObject.logTemplate)
            if len(set_seq & set_template) < 0.5 * size_seq:
                continue
            lcs = self.LCS(seq, LCSObject.logTemplate)
            if len(lcs) > maxLen or (len(lcs) == maxLen and len(LCSObject.logTemplate) < len(maxLCSObject.logTemplate)):
                maxLen = len(lcs)
                maxLCSObject = LCSObject

        # LCS should be large then tau * len(itself)
        if float(maxLen) >= self.tau * size_seq:
            retLCSObject = maxLCSObject

        return retLCSObject

    def getTemplate(self, lcs, seq):
        retVal = []
        if not lcs:
            return retVal

        lcs = lcs[::-1]
        i = 0
        for token in seq:
            i += 1
            if token == lcs[-1]:
                retVal.append(token)
                lcs.pop()
            else:
                retVal.append('<*>')
            if not lcs:
                break
        if i < len(seq):
            retVal.append('<*>')
        return retVal

    def addSeqToPrefixTree(self, rootn, newCluster):
        parentn = rootn
        seq = newCluster.logTemplate
        seq = [w for w in seq if w != '<*>']

        for i in range(len(seq)):
            tokenInSeq = seq[i]
            # Match
            if tokenInSeq in parentn.childD:
                parentn.childD[tokenInSeq].templateNo += 1
            # Do not Match
            else:
                parentn.childD[tokenInSeq] = Node(token=tokenInSeq, templateNo=1)
            parentn = parentn.childD[tokenInSeq]

        if parentn.logClust is None:
            parentn.logClust = newCluster

    def removeSeqFromPrefixTree(self, rootn, newCluster):
        parentn = rootn
        seq = newCluster.logTemplate
        seq = [w for w in seq if w != '<*>']

        for tokenInSeq in seq:
            if tokenInSeq in parentn.childD:
                matchedNode = parentn.childD[tokenInSeq]
                if matchedNode.templateNo == 1:
                    del parentn.childD[tokenInSeq]
                    break
                else:
                    matchedNode.templateNo -= 1
                    parentn = matchedNode

    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(r'\\ +', r' ', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += f'(?P<{header}>.*?)'
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def log_to_dataframe(self, log):
        """ Function to create initial dataframe
        """
        log_messages = []
        linecount = 0
        total_line = len(log)

        for line in log:
            if len(line) > self.text_max_length:
                logging.error('Length of log string is too long')
                logging.error(line)
                continue
            t = Timer(1.0, self._log_to_dataframe_handler)
            t.start()
            line = re.sub(r'[^\x00-\x7F]+', '<NASCII>', line)
            try:
                match = self.regex.search(line.strip())
                message = [match.group(header) for header in self.headers]
                log_messages.append(message)
                linecount += 1
                if linecount % 10000 == 0 or linecount == total_line:
                    logging.info('Loaded {0:.1f}% of log lines.'.format(linecount*100/total_line))
            except Exception as e:
                _ = e
                pass
            t.cancel()
        df_log = pd.DataFrame(log_messages, columns=self.headers)
        df_log.insert(0, 'LineId', None)
        df_log['LineId'] = [i + 1 for i in range(linecount)]
        return df_log

    def parseFile(self, file, persistence=True):
        starttime = datetime.now()
        filepath = os.path.join(self.path, file)
        logging.info('Parsing file: ' + filepath)
        self.logname = file
        with open(filepath, 'r') as f:
            # self.df_log = self.log_to_dataframe(f)
            self.df_log = self.log_to_dataframe(f.readlines())
        # logging.info('log_to_dataframe() finished!')
        logging.info('Pre-processing done. [Time taken: {!s}]'.format(datetime.now() - starttime))
        return self.parse(persistence)

    def parseLines(self, lines, persistence=True):
        self.logname = self.logmain
        starttime = datetime.now()
        logging.info(f'Parsing {len(lines)} lines')
        self.df_log = self.log_to_dataframe(lines)
        # logging.info('log_to_dataframe() finished!')
        logging.info('Pre-processing done. [Time taken: {!s}]'.format(datetime.now() - starttime))
        return self.parse(persistence)

    def parse(self, persistence=True):
        '''
        Function used to parse self.df_log, which is set by either parseFile() or parseLines().
        If you want to call this function manually you can call log_to_dataframe() to obtain the df and then set it with setDataframe().
        '''

        self.setLatestLineId()

        starttime = datetime.now()

        self.df_log['LineId'] = self.df_log['LineId'].apply(lambda x: x + self.lastestLineId)

        count = 0
        for _, line in self.df_log.iterrows():
            logID = line['LineId']
            logmessageL = list(filter(lambda x: x != '', re.split(r'[\s=:,]', line['Content'])))
            constLogMessL = [w for w in logmessageL if w != '<*>']

            # Find an existing matched log cluster
            matchCluster = self.PrefixTreeMatch(self.rootNode, constLogMessL, 0)

            if matchCluster is None:
                matchCluster = self.SimpleLoopMatch(self.logCluL, constLogMessL)

                if matchCluster is None:
                    matchCluster = self.LCSMatch(self.logCluL, logmessageL)

                    # Match no existing log cluster
                    if matchCluster is None:
                        newCluster = LCSObject(logTemplate=logmessageL, logIDL=[logID])
                        self.logCluL.append(newCluster)
                        self.addSeqToPrefixTree(self.rootNode, newCluster)
                    # Add the new log message to the existing cluster
                    else:
                        newTemplate = self.getTemplate(self.LCS(logmessageL, matchCluster.logTemplate),
                                                       matchCluster.logTemplate)
                        if ' '.join(newTemplate) != ' '.join(matchCluster.logTemplate):
                            self.removeSeqFromPrefixTree(self.rootNode, matchCluster)
                            matchCluster.logTemplate = newTemplate
                            self.addSeqToPrefixTree(self.rootNode, matchCluster)
            if matchCluster:
                for i in range(len(self.logCluL)):
                    if matchCluster.logTemplate == self.logCluL[i].logTemplate:
                        self.logCluL[i].logIDL.append(logID)
                        break
            count += 1
            if count % 10000 == 0 or count == len(self.df_log):
                logging.info('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log)))

        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)
        if persistence:
            self.outputResult(self.logCluL)

            if self.logmain:
                self.appendResult(self.logCluL)

        rootNodePath = os.path.join(self.savePath, 'rootNode.pkl')
        logCluLPath = os.path.join(self.savePath, 'logCluL.pkl')
        logging.info(f'rootNodePath: {rootNodePath}')
        with open(rootNodePath, 'wb') as output:
            pickle.dump(self.rootNode, output, pickle.HIGHEST_PROTOCOL)
        logging.info(f'logCluLPath: {logCluLPath}')
        with open(logCluLPath, 'wb') as output:
            pickle.dump(self.logCluL, output, pickle.HIGHEST_PROTOCOL)
        logging.info('Store objects done.')

        logging.info('Parsing done. [Time taken: {!s}]'.format(datetime.now() - starttime))

        return self.df_log

    def get_parameter_list(self, row):
        event_template = str(row["EventTemplate"])
        template_regex = re.sub(r"\s<.{1,5}>\s", "<*>", event_template)
        if "<*>" not in template_regex:
            return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
        template_regex = re.sub(r'\\ +', r'[^A-Za-z0-9]+', template_regex)
        template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
        t = Timer(1.0, self._parameter_handler)
        t.start()
        try:
            parameter_list = self._get_parameter_list(row, template_regex)
        except Exception as e:
            logging.error(e)
            parameter_list = ["TIMEOUT"]
        t.cancel()
        return parameter_list

    def _get_parameter_list(self, row, template_regex):
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        parameter_list = [para.strip(string.punctuation).strip(' ') for para in parameter_list]
        return parameter_list

    def outputResult(self, logClustL):
        if self.df_log.shape[0] == 0:
            return

        templates = [0] * self.df_log.shape[0]
        ids = [0] * self.df_log.shape[0]
        df_event = []

        for logclust in logClustL:
            template_str = ' '.join(logclust.logTemplate)
            eid = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
            for logid in logclust.logIDL:
                if logid <= self.lastestLineId:
                    continue
                templates[logid - self.lastestLineId - 1] = template_str
                ids[logid - self.lastestLineId - 1] = eid
            df_event.append([eid, template_str, len(logclust.logIDL)])

        df_event = pd.DataFrame(df_event, columns=['EventId', 'EventTemplate', 'Occurrences'])

        self.df_log['EventId'] = ids
        self.df_log['EventTemplate'] = templates
        if self.keep_para:
            self.df_log["ParameterList"] = self.df_log.apply(self.get_parameter_list, axis=1)
        logging.info('Output parse file')
        self.df_log.to_csv(os.path.join(self.savePath, self.logname + '_structured.csv'), index=False)
        df_event.to_csv(os.path.join(self.savePath, self.logname + '_templates.csv'), index=False)

        # output Main file
        if self.logmain:
            if not os.path.exists(os.path.join(self.savePath, self.logmain + '_main_structured.csv')):
                logging.info('Output main file for append')
                self.df_log.to_csv(os.path.join(self.savePath, self.logmain + '_main_structured.csv'), index=False)
                df_event.to_csv(os.path.join(self.savePath, self.logmain + '_main_templates.csv'), index=False)

    def appendResult(self, logClustL):
        if self.df_log.shape[0] == 0:
            return

        main_structured_path = os.path.join(self.savePath, self.logmain+'_main_structured.csv')
        df_log_main_structured = pd.read_csv(main_structured_path)
        lastestLineId = df_log_main_structured['LineId'].max()
        logging.info(f'lastestLindId: {lastestLineId}')

        templates = [0] * self.df_log.shape[0]
        ids = [0] * self.df_log.shape[0]
        df_event = []

        for logclust in logClustL:
            template_str = ' '.join(logclust.logTemplate)
            eid = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
            for logid in logclust.logIDL:
                if logid <= lastestLineId:
                    continue
                idx = logid - lastestLineId - 1
                templates[idx] = template_str
                ids[idx] = eid
            df_event.append([eid, template_str, len(logclust.logIDL)])

        df_event = pd.DataFrame(df_event, columns=['EventId', 'EventTemplate', 'Occurrences'])

        self.df_log['EventId'] = ids
        self.df_log['EventTemplate'] = templates
        if self.keep_para:
            self.df_log['ParameterList'] = self.df_log.apply(self.get_parameter_list, axis=1)

        df_log_append = pd.concat([df_log_main_structured, self.df_log])
        df_log_append = df_log_append[df_log_append['EventId'] != 0]
        df_log_append.to_csv(main_structured_path, index=False)
        df_event.to_csv(os.path.join(self.savePath, self.logmain + '_main_templates.csv'), index=False)

    def _parameter_handler(self):
        logging.error("_get_parameter_list function is hangs!")
        raise Exception("TIME OUT!")

    def _log_to_dataframe_handler(self):
        logging.error('log_to_dataframe function is hangs')
        raise Exception("TIME OUT!")
