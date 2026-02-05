    def _on_scene_mouse_moved(self, evt):
        """
        Handle mouse hover events on the plot scene to show tooltips or highlight points.
        The 'evt' argument is usually a tuple or a MouseEvent from the SignalProxy.
        """
        if not hasattr(self, 'plot_widget') or self.plot_widget is None:
            return

        # Check for proxy output (usually a tuple of (event,))
        if isinstance(evt, tuple):
            pos = evt[0]
        else:
            pos = evt
            
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            
            # Find nearest point (optional simple implementation)
            # For high performance with 10k+ points, KDTree is recommended.
            # Here we just pass to prevent crash.
            pass
