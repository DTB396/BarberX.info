"""
Data Visualization & Reporting API
Interactive charts, network graphs, timelines, geographic maps, PDF reports
"""
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from pyvis.network import Network
import folium
import pandas as pd
from weasyprint import HTML
from datetime import datetime, timedelta
import base64
import io

router = APIRouter(prefix="/api/v1/visualization", tags=["Visualization & Reporting"])


class TimelineEventData(BaseModel):
    timestamp: datetime
    event_type: str
    description: str
    source: Optional[str] = None
    importance: int = Field(default=3, ge=1, le=5)
    metadata: Dict[str, Any] = {}


class TimelineRequest(BaseModel):
    events: List[TimelineEventData]
    title: str = Field(default="Case Timeline")
    group_by: Optional[str] = Field(None, description="Group events by: source, type, date")
    output_format: str = Field(default="html", description="html, json, or image")


class NetworkNode(BaseModel):
    id: str
    label: str
    node_type: str  # person, organization, location, event
    properties: Dict[str, Any] = {}


class NetworkEdge(BaseModel):
    source: str
    target: str
    relationship: str
    weight: float = 1.0
    properties: Dict[str, Any] = {}


class NetworkGraphRequest(BaseModel):
    nodes: List[NetworkNode]
    edges: List[NetworkEdge]
    title: str = Field(default="Entity Relationship Graph")
    layout: str = Field(default="force", description="force, hierarchical, circular")
    output_format: str = Field(default="html")


class GeographicPoint(BaseModel):
    latitude: float
    longitude: float
    label: str
    category: str  # incident, witness, evidence, location
    details: Dict[str, Any] = {}


class GeographicMapRequest(BaseModel):
    points: List[GeographicPoint]
    center: Optional[Dict[str, float]] = None
    zoom: int = Field(default=12, ge=1, le=20)
    title: str = Field(default="Incident Map")


class ChartDataSeries(BaseModel):
    name: str
    data: List[float]
    labels: Optional[List[str]] = None


class ChartRequest(BaseModel):
    chart_type: str = Field(..., description="bar, line, pie, scatter, heatmap")
    title: str
    series: List[ChartDataSeries]
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    output_format: str = Field(default="html")


class ReportSection(BaseModel):
    title: str
    content: str
    subsections: List['ReportSection'] = []


class ReportRequest(BaseModel):
    case_name: str
    case_number: str
    prepared_by: str
    preparation_date: datetime
    sections: List[ReportSection]
    include_toc: bool = True
    include_exhibits: bool = True
    template: str = Field(default="standard", description="standard, settlement, expert")


@router.post("/timeline")
async def create_timeline(request: TimelineRequest):
    """
    Create interactive timeline visualization
    
    Perfect for:
    - Case chronologies
    - Evidence timelines
    - Incident reconstructions
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': e.timestamp,
            'event': e.description,
            'type': e.event_type,
            'source': e.source or 'Unknown',
            'importance': e.importance
        } for e in request.events])
        
        df = df.sort_values('timestamp')
        
        # Create Plotly timeline
        if request.group_by:
            # Grouped timeline
            fig = px.timeline(
                df,
                x_start='timestamp',
                x_end='timestamp',
                y=request.group_by,
                color='type',
                hover_data=['event', 'source'],
                title=request.title
            )
        else:
            # Simple timeline
            fig = go.Figure()
            
            # Color by importance
            colors = {1: '#90EE90', 2: '#FFD700', 3: '#FFA500', 4: '#FF6347', 5: '#DC143C'}
            
            for idx, row in df.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row['timestamp']],
                    y=[0],
                    mode='markers+text',
                    name=row['type'],
                    text=row['event'],
                    textposition='top center',
                    marker=dict(
                        size=10 + row['importance'] * 2,
                        color=colors.get(row['importance'], '#808080')
                    ),
                    hovertext=f"{row['event']}<br>Source: {row['source']}"
                ))
            
            fig.update_layout(
                title=request.title,
                xaxis_title='Date/Time',
                yaxis=dict(visible=False),
                showlegend=True,
                height=400
            )
        
        if request.output_format == 'html':
            return Response(content=fig.to_html(), media_type='text/html')
        elif request.output_format == 'json':
            return fig.to_json()
        elif request.output_format == 'image':
            img_bytes = fig.to_image(format='png')
            return Response(content=img_bytes, media_type='image/png')
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Timeline creation failed: {str(e)}")


@router.post("/network-graph")
async def create_network_graph(request: NetworkGraphRequest):
    """
    Create entity relationship network graph
    
    Visualize connections between:
    - People, organizations, locations
    - Events and evidence
    - Complex relationship patterns
    """
    try:
        # Create NetworkX graph
        G = nx.DiGraph() if any(e.properties.get('directed') for e in request.edges) else nx.Graph()
        
        # Add nodes
        for node in request.nodes:
            G.add_node(
                node.id,
                label=node.label,
                type=node.node_type,
                **node.properties
            )
        
        # Add edges
        for edge in request.edges:
            G.add_edge(
                edge.source,
                edge.target,
                relationship=edge.relationship,
                weight=edge.weight,
                **edge.properties
            )
        
        # Create interactive visualization with pyvis
        net = Network(height='750px', width='100%', directed=G.is_directed())
        
        # Color nodes by type
        node_colors = {
            'person': '#FF6B6B',
            'organization': '#4ECDC4',
            'location': '#95E1D3',
            'event': '#FFD93D',
            'evidence': '#6C5CE7'
        }
        
        for node in request.nodes:
            net.add_node(
                node.id,
                label=node.label,
                title=f"{node.node_type}: {node.label}",
                color=node_colors.get(node.node_type, '#808080')
            )
        
        for edge in request.edges:
            net.add_edge(
                edge.source,
                edge.target,
                title=edge.relationship,
                width=edge.weight
            )
        
        # Set layout
        if request.layout == 'hierarchical':
            net.set_options("""
            var options = {
                "layout": {
                    "hierarchical": {
                        "enabled": true,
                        "direction": "UD",
                        "sortMethod": "directed"
                    }
                }
            }
            """)
        elif request.layout == 'circular':
            net.toggle_physics(False)
        else:
            net.barnes_hut()
        
        # Generate HTML
        html = net.generate_html()
        
        if request.output_format == 'html':
            return Response(content=html, media_type='text/html')
        elif request.output_format == 'json':
            return {
                'nodes': [{'id': n.id, 'label': n.label, 'type': n.node_type} for n in request.nodes],
                'edges': [{'source': e.source, 'target': e.target, 'relationship': e.relationship} for e in request.edges]
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Network graph creation failed: {str(e)}")


@router.post("/geographic-map")
async def create_geographic_map(request: GeographicMapRequest):
    """
    Create interactive geographic map
    
    Plot incident locations, evidence sites, witness locations
    """
    try:
        # Determine center point
        if request.center:
            center = [request.center['latitude'], request.center['longitude']]
        else:
            # Calculate center from points
            avg_lat = sum(p.latitude for p in request.points) / len(request.points)
            avg_lon = sum(p.longitude for p in request.points) / len(request.points)
            center = [avg_lat, avg_lon]
        
        # Create map
        m = folium.Map(location=center, zoom_start=request.zoom)
        
        # Add markers
        marker_colors = {
            'incident': 'red',
            'witness': 'blue',
            'evidence': 'green',
            'location': 'gray'
        }
        
        marker_icons = {
            'incident': 'exclamation-circle',
            'witness': 'user',
            'evidence': 'camera',
            'location': 'map-marker'
        }
        
        for point in request.points:
            # Create popup content
            popup_html = f"""
            <div style="width: 200px">
                <h4>{point.label}</h4>
                <p><strong>Category:</strong> {point.category}</p>
                {''.join(f'<p><strong>{k}:</strong> {v}</p>' for k, v in point.details.items())}
            </div>
            """
            
            folium.Marker(
                location=[point.latitude, point.longitude],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=point.label,
                icon=folium.Icon(
                    color=marker_colors.get(point.category, 'gray'),
                    icon=marker_icons.get(point.category, 'info-sign')
                )
            ).add_to(m)
        
        # Add title
        title_html = f'''
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 300px; height: 50px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:16px; font-weight: bold; padding: 10px">
            {request.title}
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Return HTML
        return Response(content=m._repr_html_(), media_type='text/html')
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Map creation failed: {str(e)}")


@router.post("/chart")
async def create_chart(request: ChartRequest):
    """
    Create various chart types
    
    Supports: bar, line, pie, scatter, heatmap
    """
    try:
        fig = None
        
        if request.chart_type == 'bar':
            fig = go.Figure()
            for series in request.series:
                fig.add_trace(go.Bar(
                    name=series.name,
                    x=series.labels or list(range(len(series.data))),
                    y=series.data
                ))
            fig.update_layout(barmode='group')
            
        elif request.chart_type == 'line':
            fig = go.Figure()
            for series in request.series:
                fig.add_trace(go.Scatter(
                    name=series.name,
                    x=series.labels or list(range(len(series.data))),
                    y=series.data,
                    mode='lines+markers'
                ))
                
        elif request.chart_type == 'pie':
            # Use first series for pie chart
            series = request.series[0]
            fig = go.Figure(data=[go.Pie(
                labels=series.labels or [f'Item {i+1}' for i in range(len(series.data))],
                values=series.data,
                name=series.name
            )])
            
        elif request.chart_type == 'scatter':
            fig = go.Figure()
            for i in range(0, len(request.series), 2):
                if i+1 < len(request.series):
                    fig.add_trace(go.Scatter(
                        x=request.series[i].data,
                        y=request.series[i+1].data,
                        mode='markers',
                        name=f'{request.series[i].name} vs {request.series[i+1].name}'
                    ))
        
        if fig:
            fig.update_layout(
                title=request.title,
                xaxis_title=request.x_label,
                yaxis_title=request.y_label
            )
            
            if request.output_format == 'html':
                return Response(content=fig.to_html(), media_type='text/html')
            elif request.output_format == 'json':
                return fig.to_json()
            elif request.output_format == 'image':
                img_bytes = fig.to_image(format='png')
                return Response(content=img_bytes, media_type='image/png')
        
        raise HTTPException(status_code=400, detail=f"Unsupported chart type: {request.chart_type}")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart creation failed: {str(e)}")


@router.post("/generate-report")
async def generate_report(request: ReportRequest):
    """
    Generate professional PDF report
    
    Features:
    - Table of contents
    - Section headers
    - Embedded charts and images
    - Court-ready formatting
    """
    try:
        # Build HTML for report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                @page {{
                    size: letter;
                    margin: 1in;
                    @top-center {{
                        content: "{request.case_name}";
                    }}
                    @bottom-center {{
                        content: "Page " counter(page) " of " counter(pages);
                    }}
                }}
                body {{
                    font-family: 'Times New Roman', serif;
                    font-size: 12pt;
                    line-height: 1.6;
                }}
                h1 {{
                    font-size: 18pt;
                    font-weight: bold;
                    margin-top: 24pt;
                    margin-bottom: 12pt;
                    text-align: center;
                }}
                h2 {{
                    font-size: 14pt;
                    font-weight: bold;
                    margin-top: 18pt;
                    margin-bottom: 6pt;
                }}
                h3 {{
                    font-size: 12pt;
                    font-weight: bold;
                    margin-top: 12pt;
                    margin-bottom: 6pt;
                }}
                .cover-page {{
                    page-break-after: always;
                    text-align: center;
                    padding-top: 3in;
                }}
                .section {{
                    margin-bottom: 24pt;
                }}
                .toc {{
                    page-break-after: always;
                }}
                .page-break {{
                    page-break-before: always;
                }}
            </style>
        </head>
        <body>
            <!-- Cover Page -->
            <div class="cover-page">
                <h1>{request.case_name}</h1>
                <p><strong>Case Number:</strong> {request.case_number}</p>
                <p><strong>Prepared By:</strong> {request.prepared_by}</p>
                <p><strong>Date:</strong> {request.preparation_date.strftime('%B %d, %Y')}</p>
            </div>
        """
        
        # Table of Contents
        if request.include_toc:
            html_content += """
            <div class="toc">
                <h2>Table of Contents</h2>
                <ol>
            """
            for i, section in enumerate(request.sections):
                html_content += f"<li>{section.title}</li>"
            html_content += """
                </ol>
            </div>
            """
        
        # Sections
        for section in request.sections:
            html_content += f"""
            <div class="section">
                <h2>{section.title}</h2>
                <p>{section.content}</p>
            """
            
            # Subsections
            for subsection in section.subsections:
                html_content += f"""
                <h3>{subsection.title}</h3>
                <p>{subsection.content}</p>
                """
            
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        # Generate PDF
        pdf_bytes = HTML(string=html_content).write_pdf()
        
        return Response(
            content=pdf_bytes,
            media_type='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="{request.case_name.replace(" ", "_")}_Report.pdf"'
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.get("/analytics-dashboard/{case_id}")
async def get_analytics_dashboard(case_id: str):
    """
    Generate comprehensive analytics dashboard for a case
    
    Includes:
    - Document statistics
    - Timeline visualization
    - Entity network graph
    - Key metrics
    """
    # TODO: Fetch case data and generate dashboard
    return {"message": "Dashboard generation - implementation pending"}
