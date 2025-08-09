import os
import paramiko

class sftpClient():
    def __init__(self):
        self.host = '127.0.0.1'
        self.port = 22
        self.user = 'admin'
        self.passwd = ''
        self.rootDir = '~/data'
        self.sftp = None
        try:
            self.transport = paramiko.Transport((self.host, self.port))
            self.transport.connect(username=self.user, password=self.passwd)
            self.sftp = paramiko.SFTPClient.from_transport(self.transport)

        except Exception as e:
            print('SFTP ERROR:' + str(e) + '      Will IGNORE')

    def __del__(self):
        if self.sftp is not None:
            self.sftp.close()
            self.transport.close()

    def createFolderIfNotExist(self, remote_directory):
        if self.sftp is None:
            return
        remote_full_directory = os.path.join(self.rootDir, remote_directory)
        directories = remote_full_directory.split('/')
        path = ''
        for directory in directories:
            if directory:
                path += '/' + directory
                try:
                    self.sftp.stat(path)
                except FileNotFoundError:
                    self.sftp.mkdir(path)



    def save(self, local_file_path):
        if self.sftp is not None:
            remote_file_path = os.path.join(self.rootDir, local_file_path)
            try:
                self.sftp.put(local_file_path, remote_file_path)
                print('save file form {} to {}:{}-{}'.format(local_file_path, self.host, self.port, remote_file_path))

            except Exception as e:
                print(str(e))

    def remove_dir(self, remote_dir):
        if self.sftp is not None:
            self.sftp.remove(remote_dir+'/*')
            self.sftp.rmdir(remote_dir)


