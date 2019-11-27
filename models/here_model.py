from torchvision import models
import torch.nn as nn
import torch
    
class HereModel(nn.Module):
    def __init__(self):
        super(HereModel, self).__init__()
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
        cnn_front = models.resnet50(pretrained=True)
        self.front_features = nn.Sequential(*list(cnn_front.children())[:-1])
        self.front_intermediate = nn.Sequential(
            nn.Linear(cnn_front.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        final_concat_size += 128

        # Main LSTM
        self.front_lstm = nn.LSTM(input_size=128,
            hidden_size=64,
            num_layers=3,
            batch_first=False)
        final_concat_size += 64
        
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
        lstm_i = []
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
        for k, v in data['cameraFront'].items():
            v = v.cuda()
            x = self.front_features(v)
            x = x.view(x.size(0), -1)
            x = self.front_intermediate(x)
            lstm_i.append(x)
            # feed the current front facing camera
            # output directly into the 
            # regression networks.
            if k == 0:
                module_outputs.append(x)

        # Feed temporal outputs of CNN into LSTM
        i_lstm, _ = self.front_lstm(torch.stack(lstm_i))
        module_outputs.append(i_lstm[-1])
        
        # Concatenate current image CNN output 
        # and LSTM output.
        x_cat = torch.cat(module_outputs, dim=-1)
        
        # Feed concatenated outputs into the 
        # regession networks.
        prediction = {'canSteering': torch.squeeze(self.control_angle(x_cat)),
                      'canSpeed': torch.squeeze(self.control_speed(x_cat))}
        return prediction