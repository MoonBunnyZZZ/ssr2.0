stream1_cfg = dict(backbone=dict(stem=dict(out_channels=32, kernel_size=5, stride=2, padding=4),
                                 stage1=dict(in_chs=32, out_chs=32, stride=1, depth=2),
                                 stage2=dict(in_chs=32, out_chs=64, stride=2, depth=2),
                                 stage3=dict(in_chs=64, out_chs=128, stride=2, depth=2),
                                 stage4=dict(in_chs=128, out_chs=256, stride=2, depth=2)),
                   head=dict(head1=64, head2=128, head3=256))

stream2_cfg = dict(backbone=dict(stem=dict(out_channels=16, kernel_size=5, stride=2, padding=4),
                                 stage1=dict(in_chs=16, out_chs=16, stride=1, depth=2),
                                 stage2=dict(in_chs=16, out_chs=32, stride=2, depth=2),
                                 stage3=dict(in_chs=32, out_chs=64, stride=2, depth=2),
                                 stage4=dict(in_chs=64, out_chs=128, stride=2, depth=2)),
                   head=dict(head1=32, head2=64, head3=128))
