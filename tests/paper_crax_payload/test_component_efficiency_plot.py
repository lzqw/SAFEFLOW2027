"""A synthetic plotting-only check: no policies, training or simulation."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.paper_crax_payload.report_component_study import plot_endpoint_efficiency


def test_equal_risk_different_capped_times_are_not_collapsed(tmp_path, monkeypatch):
    rows=[dict(arm=name,condition=condition,violation=0.,capped_seconds=seconds)
          for condition in ('nominal','radius065','execution085')
          for name,seconds in (('metric-only',.71),('full',.73))]
    saved=[]
    from matplotlib.figure import Figure
    original=Figure.savefig

    def capture(figure, *args, **kwargs):
        if not saved:
            saved.append([[collection.get_offsets().tolist() for collection in ax.collections]
                          for ax in figure.axes])
        return original(figure,*args,**kwargs)

    monkeypatch.setattr(Figure,'savefig',capture)
    plot_endpoint_efficiency(rows,['metric-only','full'],tmp_path)
    assert saved[0] == [[[[0.,.71]],[[0.,.73]]]] * 3
    assert (tmp_path/'fixed_endpoint_risk_efficiency.png').stat().st_size > 0
    assert (tmp_path/'fixed_endpoint_risk_efficiency.pdf').stat().st_size > 0
    assert not plt.get_fignums()
