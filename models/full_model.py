from torchvision import models
import torch.nn as nn
import torch

class FullModel(nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()
        # For number inputs
        self.number_labels = ['hereSpeedLimit',
                                'hereFreeFlowSpeed', 'hereSignal', 'hereYield',
                               'herePedestrian', 'hereIntersection', 'hereMmIntersection',
                               'hereSegmentExitHeading', 'hereSegmentEntryHeading',
                               'hereCurvature', 
                               'here1mHeading', 'here5mHeading', 'here10mHeading', 'here20mHeading',
                               'here50mHeading']
        final_concat_size = len(self.number_labels)

        
        # CNN Front
        front_cnn = models.resnet50(pretrained=True)
        self.front_features = nn.Sequential(*list(front_cnn.children())[:-1])
        self.front_intermediate = nn.Sequential(
            nn.Linear(front_cnn.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        final_concat_size += 128
        
        # CNN Left
        left_cnn = models.resnet34(pretrained=True)
        self.left_features = nn.Sequential(*list(left_cnn.children())[:-1])
        self.left_intermediate = nn.Sequential(
            nn.Linear(left_cnn.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        final_concat_size += 128
        
        # CNN Right
        right_cnn = models.resnet34(pretrained=True)
        self.right_features = nn.Sequential(*list(right_cnn.children())[:-1])
        self.right_intermediate = nn.Sequential(
            nn.Linear(right_cnn.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        final_concat_size += 128
        
        # CNN Rear
#        cnn["cameraRear"] = models.resnet50(pretrained=True)
#        self.cnn_features["cameraRear"] = nn.Sequential(*list(cnn["cameraRear"].children())[:-1])
#        self.cnn_intermediate["cameraRear"] = nn.Sequential(
#            nn.Linear(cnn["cameraRear"].fc.in_features, 256),
#            nn.ReLU(),
#            nn.Dropout(0.2),
#            nn.Linear(256, 128),
#            nn.ReLU(),
#            nn.Dropout(0.5)
#        )
#        final_concat_size += 128

        # Main LSTM
        self.lstm_front = nn.LSTM(input_size=128,
            hidden_size=64,
            num_layers=3,
            batch_first=False)
        final_concat_size += 64
        
#        self.lstm["cameraRear"] = nn.LSTM(input_size=128,
#            hidden_size=64,
#            num_layers=3,
#            batch_first=False)
#        final_concat_size += 64
        
        # CNN HERE
        cnn_here = models.resnet34(pretrained=True)
        self.here_features = nn.Sequential(*list(cnn_here.children())[:-1])
        self.here_intermediate = nn.Sequential(nn.Linear(
                          cnn_here.fc.in_features, 128),
                          nn.ReLU())
        final_concat_size += 128
        
        # Angle Regressor
        self.control_angle = nn.Sequential(
            nn.Linear(final_concat_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )
        # Speed Regressor
        self.control_speed = nn.Sequential(
            nn.Linear(final_concat_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )
    
    def forward(self, data):
        module_outputs = []
        # Number parameters
        for numbers in self.number_labels:
            module_outputs.append(data[numbers].float().cuda())
            
        # HERE map
        here = self.here_features(data['here'].cuda())
        here = here.view(here.size(0), -1)
        here = self.here_intermediate(here)
        module_outputs.append(here)
        
        # Loop through temporal sequence of
        # front facing camera images and pass 
        # through the cnn.
        # keys = ['cameraFront', 'cameraLeft', 'cameraRight', 'cameraRear']
        
        lstm_i_front = []
        for k, v in data['cameraFront'].items():
            v = v.cuda()
            x = self.front_features(v)
            x = x.view(x.size(0), -1)
            x = self.front_intermediate(x)
            lstm_i_front.append(x)
            # feed the current front facing camera
            # output directly into the 
            # regression networks.
            if k == 0:
                module_outputs.append(x)
        # Feed temporal outputs of CNN into LSTM
        lstm_i_front, _ = self.lstm_front(torch.stack(lstm_i_front))
        module_outputs.append(lstm_i_front[-1])
        
        left = self.left_features(data['cameraLeft'].cuda())
        left = left.view(left.size(0), -1)
        left = self.left_intermediate(left)
        module_outputs.append(left)
        
        right = self.right_features(data['cameraRight'].cuda())
        right = right.view(right.size(0), -1)
        right = self.right_intermediate(right)
        module_outputs.append(right)
        
        # Concatenate current image CNN output 
        # and LSTM output.
        x_cat = torch.cat(module_outputs, dim=-1)
        
        # Feed concatenated outputs into the 
        # regession networks.
        prediction = {'canSteering': torch.squeeze(self.control_angle(x_cat)),
                      'canSpeed': torch.squeeze(self.control_speed(x_cat))}
        return prediction