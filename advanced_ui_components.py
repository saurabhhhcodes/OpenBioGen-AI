"""
Advanced UI Components for OpenBioGen AI
Enhanced Streamlit components with modern design and functionality
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

class AdvancedUIComponents:
    """Advanced UI components for enhanced user experience"""
    
    @staticmethod
    def create_metric_card(title: str, value: str, delta: str = None, color: str = "blue"):
        """Create an enhanced metric card"""
        color_map = {
            "blue": "#1f77b4",
            "green": "#2ca02c", 
            "red": "#d62728",
            "orange": "#ff7f0e",
            "purple": "#9467bd"
        }
        
        card_color = color_map.get(color, "#1f77b4")
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {card_color}15, {card_color}05);
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid {card_color};
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <h4 style="color: {card_color}; margin: 0; font-size: 0.9rem;">{title}</h4>
            <h2 style="margin: 0.5rem 0; color: #333;">{value}</h2>
            {f'<p style="color: #666; margin: 0; font-size: 0.8rem;">{delta}</p>' if delta else ''}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_progress_ring(percentage: float, title: str, color: str = "#1f77b4"):
        """Create a circular progress indicator"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title, 'font': {'size': 16}},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 50], 'color': "gray"},
                    {'range': [50, 75], 'color': "lightblue"},
                    {'range': [75, 100], 'color': "blue"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        return fig
    
    @staticmethod
    def create_confidence_visualization(confidence_data: List[Dict[str, Any]]):
        """Create advanced confidence score visualization"""
        if not confidence_data:
            return None
        
        df = pd.DataFrame(confidence_data)
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Confidence Distribution', 'Risk Categories', 'Timeline', 'Gene Frequency'),
            specs=[[{"secondary_y": False}, {"type": "pie"}],
                   [{"secondary_y": True}, {"type": "bar"}]]
        )
        
        # Confidence distribution histogram
        fig.add_trace(
            go.Histogram(x=df['confidence_score'], nbinsx=20, name="Confidence"),
            row=1, col=1
        )
        
        # Risk categories pie chart
        risk_counts = df['risk_category'].value_counts()
        fig.add_trace(
            go.Pie(labels=risk_counts.index, values=risk_counts.values, name="Risk Categories"),
            row=1, col=2
        )
        
        # Timeline of predictions
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        timeline_data = df.groupby(df['timestamp'].dt.date).size()
        fig.add_trace(
            go.Scatter(x=timeline_data.index, y=timeline_data.values, mode='lines+markers', name="Predictions"),
            row=2, col=1
        )
        
        # Gene frequency
        gene_counts = df['gene'].value_counts().head(10)
        fig.add_trace(
            go.Bar(x=gene_counts.index, y=gene_counts.values, name="Gene Frequency"),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Comprehensive Analysis Dashboard"
        )
        
        return fig
    
    @staticmethod
    def create_risk_heatmap(risk_data: Dict[str, Dict[str, float]]):
        """Create risk assessment heatmap"""
        if not risk_data:
            return None
        
        genes = list(risk_data.keys())
        diseases = list(set().union(*[d.keys() for d in risk_data.values()]))
        
        z_data = []
        for disease in diseases:
            row = []
            for gene in genes:
                risk_score = risk_data.get(gene, {}).get(disease, 0)
                row.append(risk_score)
            z_data.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=genes,
            y=diseases,
            colorscale='RdYlBu_r',
            hoverongaps=False,
            colorbar=dict(title="Risk Score")
        ))
        
        fig.update_layout(
            title="Gene-Disease Risk Assessment Heatmap",
            xaxis_title="Genes",
            yaxis_title="Diseases",
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_interactive_network(network_data: Dict[str, Any]):
        """Create interactive network visualization"""
        if not network_data or 'nodes' not in network_data:
            return None
        
        nodes = network_data['nodes']
        edges = network_data.get('edges', [])
        
        # Create network layout
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for i, node in enumerate(nodes):
            # Simple circular layout
            angle = 2 * np.pi * i / len(nodes)
            x = np.cos(angle)
            y = np.sin(angle)
            
            node_x.append(x)
            node_y.append(y)
            node_text.append(node['name'])
            node_color.append(node.get('score', 0.5))
        
        # Create edges
        edge_x = []
        edge_y = []
        
        for edge in edges:
            source_idx = next((i for i, n in enumerate(nodes) if n['name'] == edge['source']), 0)
            target_idx = next((i for i, n in enumerate(nodes) if n['name'] == edge['target']), 0)
            
            edge_x.extend([node_x[source_idx], node_x[target_idx], None])
            edge_y.extend([node_y[source_idx], node_y[target_idx], None])
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Connections'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=20,
                color=node_color,
                colorscale='Viridis',
                line=dict(width=2, color='white')
            ),
            name='Genes/Proteins'
        ))
        
        fig.update_layout(
            title="Gene-Protein Interaction Network",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text="Interactive network showing gene-protein interactions",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color='#888', size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_system_status_dashboard(health_data: Dict[str, Any], performance_data: Dict[str, Any]):
        """Create system status dashboard"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_color = "green" if health_data.get('overall_status') == 'healthy' else "orange"
            AdvancedUIComponents.create_metric_card(
                "System Status", 
                health_data.get('overall_status', 'Unknown').title(),
                color=status_color
            )
        
        with col2:
            cache_stats = performance_data.get('cache_stats', {})
            hit_rate = cache_stats.get('hit_rate', 0) * 100
            AdvancedUIComponents.create_metric_card(
                "Cache Hit Rate",
                f"{hit_rate:.1f}%",
                f"{cache_stats.get('hits', 0)} hits",
                color="blue"
            )
        
        with col3:
            error_count = performance_data.get('error_count', 0)
            error_color = "green" if error_count == 0 else "red" if error_count > 10 else "orange"
            AdvancedUIComponents.create_metric_card(
                "Error Count",
                str(error_count),
                "Last 24h",
                color=error_color
            )
        
        with col4:
            avg_response = performance_data.get('average_response_time', 0)
            response_color = "green" if avg_response < 1 else "orange" if avg_response < 3 else "red"
            AdvancedUIComponents.create_metric_card(
                "Avg Response",
                f"{avg_response:.2f}s",
                "Response time",
                color=response_color
            )
    
    @staticmethod
    def create_export_options(data: Any, filename_prefix: str = "openbio_export"):
        """Create advanced export options"""
        st.subheader("📤 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export as JSON", key=f"json_{filename_prefix}"):
                json_str = json.dumps(data, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if isinstance(data, (list, dict)) and st.button("📊 Export as CSV", key=f"csv_{filename_prefix}"):
                if isinstance(data, list) and data:
                    df = pd.DataFrame(data)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        with col3:
            if st.button("📋 Copy to Clipboard", key=f"copy_{filename_prefix}"):
                st.code(json.dumps(data, indent=2, default=str))
                st.success("Data displayed above - copy manually")

class AdvancedFilters:
    """Advanced filtering and search components"""
    
    @staticmethod
    def create_gene_filter(genes: List[str]) -> List[str]:
        """Create advanced gene filter"""
        st.subheader("🧬 Gene Filter")
        
        # Search box
        search_term = st.text_input("Search genes:", placeholder="Type to search...")
        
        # Filter genes based on search
        filtered_genes = [g for g in genes if search_term.lower() in g.lower()] if search_term else genes
        
        # Multi-select with search
        selected_genes = st.multiselect(
            "Select genes:",
            options=filtered_genes,
            default=filtered_genes[:5] if len(filtered_genes) > 5 else filtered_genes
        )
        
        return selected_genes
    
    @staticmethod
    def create_confidence_filter() -> tuple:
        """Create confidence score filter"""
        st.subheader("📊 Confidence Filter")
        
        confidence_range = st.slider(
            "Confidence Score Range:",
            min_value=0.0,
            max_value=1.0,
            value=(0.5, 1.0),
            step=0.1
        )
        
        confidence_categories = st.multiselect(
            "Confidence Categories:",
            options=["High", "Medium", "Low"],
            default=["High", "Medium"]
        )
        
        return confidence_range, confidence_categories
    
    @staticmethod
    def create_date_filter() -> tuple:
        """Create date range filter"""
        st.subheader("📅 Date Filter")
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date:",
                value=datetime.now() - timedelta(days=30)
            )
        
        with col2:
            end_date = st.date_input(
                "End Date:",
                value=datetime.now()
            )
        
        return start_date, end_date
