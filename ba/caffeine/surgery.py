def convert_to_FCN(new_net, old_net, new_params, old_params):
    fc_params = {pr: (old_net.params[pr][0].data,
                      old_net.params[pr][1].data) for pr in old_params}
    for fc in old_params:
        print('''{} weights are {} dimensional and
              biases are {} dimensional'''.format(fc, fc_params[fc][0].shape,
                                                  fc_params[fc][1].shape))

    conv_params = {pr: (new_net.params[pr][0].data,
                        new_net.params[pr][1].data) for pr in new_params}
    for conv in new_params:
        print('''{} weights are {} dimensional and biases are {}
              dimensional'''.format(conv, conv_params[conv][0].shape,
                                    conv_params[conv][1].shape))

    for pr, pr_conv in zip(old_params, new_params):
        conv_params[pr_conv][0].flat = fc_params[pr][0].flat
        conv_params[pr_conv][1][...] = fc_params[pr][1]
    return new_net
