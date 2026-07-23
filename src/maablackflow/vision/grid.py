"""Road evidence extraction and robust global periodic-grid fitting."""
from __future__ import annotations
from dataclasses import dataclass, field
import cv2
import numpy as np

@dataclass(frozen=True, slots=True)
class MapViewport:
    left: int; top: int; right: int; bottom: int
    right_panel_detected: bool = False
    excluded_rectangles: tuple[tuple[int,int,int,int], ...] = ()
    def contains(self, x: int, y: int) -> bool:
        return (self.left <= x < self.right and self.top <= y < self.bottom
                and not any(x1 <= x < x2 and y1 <= y < y2 for x1,y1,x2,y2 in self.excluded_rectangles))

@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """A local observation; never a final node by itself."""
    x: int; y: int; source: str; confidence: float; radius: int = 0
    scores: dict[str,float] = field(default_factory=dict)
    bbox: tuple[int, int, int, int] | None = None

@dataclass(frozen=True, slots=True)
class GridFit:
    status: str
    spacing_x: float = 0.; spacing_y: float = 0.
    origin_x: float = 0.; origin_y: float = 0.
    rows: tuple[int,...] = (); columns: tuple[int,...] = ()
    row_indices: tuple[int,...] = (); column_indices: tuple[int,...] = ()
    score_x: float = 0.; score_y: float = 0.
    @property
    def succeeded(self) -> bool: return self.status == "ok"

@dataclass(slots=True)
class RoadGridAnalysis:
    viewport: MapViewport; road_mask: np.ndarray; skeleton: np.ndarray
    local_rows: tuple[int,...]; local_columns: tuple[int,...]
    evidence: tuple[CandidateEvidence,...]; grid: GridFit
    @property
    def rows(self): return self.grid.rows
    @property
    def columns(self): return self.grid.columns
    @property
    def candidates(self): return self.evidence

def find_map_viewport(image: np.ndarray) -> MapViewport:
    h,w=image.shape[:2]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(cv2.GaussianBlur(gray,(7,7),1.5),60,140)
    left,top,right,bottom=round(w*.015),round(h*.055),round(w*.985),round(h*.86)
    rp=edges[round(h*.12):round(h*.82),round(w*.66):round(w*.97)]
    mid=edges[round(h*.12):round(h*.82),round(w*.36):round(w*.60)]
    edge_support=edges[round(h*.08):bottom,round(w*.85):round(w*.88)]
    rd,md=float(np.mean(rp>0)),float(np.mean(mid>0))
    panel=rd>.045 and rd>1.35*max(md,.001)
    if panel:
        right=round(w*.66)
    else:
        # Preserve a supported map edge while excluding the far-right controls.
        right=round(w*(.88 if float(np.mean(edge_support>0))>.04 else .87))
    excluded=[(0,round(h*.39),round(w*.065),bottom)]
    # The action-point/resource widget occupies the upper-right UI band. It is
    # a true occluder: periodic axes may cross it, but no cell is emitted there.
    if right>round(w*.80):
        excluded.append((round(w*.80),0,right,round(h*.22)))
    return MapViewport(left,top,right,bottom,panel,tuple(excluded))

def analyze_road_grid(image: np.ndarray, visual_points: tuple[tuple[int,int],...]=(),
                      viewport: MapViewport|None=None,
                      visual_evidence: tuple[CandidateEvidence,...]=()) -> RoadGridAnalysis:
    h,w=image.shape[:2]; viewport=viewport or find_map_viewport(image)
    road=_road_mask(image,viewport); skeleton=_thin(road)
    local_rows,local_cols=_line_axes(road,h,viewport)
    evidence=list(visual_evidence)
    evidence += [CandidateEvidence(x,y,s,.34,0,{"road_keypoint":.34})
                 for x,y,s in _skeleton_keypoints(skeleton,viewport,h)]
    evidence += [CandidateEvidence(x,y,"visual_anchor",.72,0,{"visual_anchor":.72}) for x,y in visual_points]
    # Local road turns/endpoints remain debug evidence, but they must not vote as
    # independent global axes. Cluster visual anchors in 2-D first so a large
    # icon producing several Hough circles gets only one lattice vote.
    anchor_evidence=[e for e in visual_evidence if e.source!="white_person_component" and e.confidence>=.68]
    anchor_evidence += [CandidateEvidence(x,y,"visual_anchor",.72) for x,y in visual_points]
    clustered_visual=_cluster_visual_anchors(anchor_evidence,max(18.,h*.05))
    xs=[(float(x),weight) for x,y,weight in clustered_visual]
    ys=[(float(y),weight) for x,y,weight in clustered_visual]
    xs += [(float(v),.72) for v in local_cols]; ys += [(float(v),.72) for v in local_rows]
    lo=max(48.,h*.105); hi=min(w*.30,h*.50)
    xf=_fit_periodic_axis(xs,lo,hi)
    yf=_fit_periodic_axis(ys,lo,hi,xf[0] if xf else None)
    if xf is None or yf is None: grid=GridFit("grid_fit_failed")
    else:
        sx,ox,qx=xf; sy,oy,qy=yf
        cols,ci=_axis_sequence(ox,sx,viewport.left,viewport.right)
        rows,ri=_axis_sequence(oy,sy,viewport.top,viewport.bottom)
        grid=(GridFit("ok",sx,sy,ox,oy,rows,cols,ri,ci,qx,qy)
              if len(cols)>=2 and len(rows)>=2 else GridFit("grid_fit_failed"))
    if grid.succeeded:
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        for row in grid.rows:
            for col in grid.columns:
                if not viewport.contains(col,row): continue
                rs,dirs=road_connectivity(road,col,row,h); ds=dark_center_score(gray,col,row,h)
                if rs<.28 and ds<.42: continue
                scores={"road":rs,"dark_center":ds,"grid":1.}
                src=["grid_road_intersection"]
                if len(dirs)>=2: src.append("road_connectivity")
                if ds>=.42: src.append("dark_center")
                evidence.append(CandidateEvidence(col,row,"+".join(src),
                    float(np.clip(.48*rs+.32*ds+.2,0,1)),0,scores))
    return RoadGridAnalysis(viewport,road,skeleton,tuple(local_rows),tuple(local_cols),tuple(evidence),grid)

def _cluster_visual_anchors(evidence, distance):
    groups=[]
    for item in sorted(evidence,key=lambda e:(-e.confidence,e.y,e.x)):
        match=None
        for group in groups:
            weights=[e.confidence for e in group]
            cx=float(np.average([e.x for e in group],weights=weights)); cy=float(np.average([e.y for e in group],weights=weights))
            if (item.x-cx)**2+(item.y-cy)**2<=distance**2: match=group; break
        if match is None: groups.append([item])
        else: match.append(item)
    result=[]
    for group in groups:
        weights=[e.confidence for e in group]
        result.append((float(np.average([e.x for e in group],weights=weights)),float(np.average([e.y for e in group],weights=weights)),min(2.,sum(weights))))
    return result
def _cluster_axis(anchors, tolerance=10.):
    groups=[]
    for value,weight in sorted(anchors):
        if groups:
            c=float(np.average([v for v,_ in groups[-1]],weights=[w for _,w in groups[-1]]))
            if abs(value-c)<=tolerance: groups[-1].append((value,weight)); continue
        groups.append([(value,weight)])
    return [(float(np.average([v for v,_ in g],weights=[w for _,w in g])),min(2.,sum(w for _,w in g))) for g in groups]

def _fit_periodic_axis(anchors, minimum_spacing, maximum_spacing, expected_spacing=None):
    clustered=_cluster_axis(anchors)
    if len(clustered)<(2 if expected_spacing is not None else 3): return None
    periods=set()
    for i,(left,_) in enumerate(clustered):
        for right,_ in clustered[i+1:]:
            for steps in range(1,7):
                s=(right-left)/steps
                if minimum_spacing<=s<=maximum_spacing: periods.add(round(s,1))
    if expected_spacing is not None:
        periods={s for s in periods if .55*expected_spacing<=s<=1.30*expected_spacing}
    total=sum(w for _,w in clustered); best=None
    minimum_indices=2 if expected_spacing is not None else 3
    for spacing in sorted(periods):
        tol=max(8.,spacing*.085)
        for anchor,_ in clustered:
            origin=anchor%spacing; iw=rw=0.; indices=set()
            for value,weight in clustered:
                idx=round((value-origin)/spacing); residual=abs(value-(origin+idx*spacing))
                if residual<=tol:
                    iw+=weight; rw+=weight*residual; indices.add(idx)
            if len(indices)<minimum_indices: continue
            mean=rw/max(iw,1e-6); coverage=iw/max(total,1e-6)
            objective=iw+.18*len(indices)-.08*mean+spacing*.006
            if expected_spacing is not None: objective-=30.0*abs(spacing-expected_spacing)/expected_spacing
            candidate=(objective,coverage,-mean,spacing,origin)
            if best is None or candidate>best: best=candidate
    if best is None or best[1]<.22: return None
    _,coverage,neg,spacing,origin=best
    quality=float(np.clip(.65*coverage+.35*max(0.,1.+neg/max(8.,spacing*.085)),0,1))
    return spacing,origin,quality

def _axis_sequence(origin,spacing,lower,upper):
    first=int(np.ceil((lower-origin)/spacing)); last=int(np.floor((upper-1-origin)/spacing))
    indices=tuple(range(first,last+1)); return tuple(round(origin+i*spacing) for i in indices),indices

def _road_mask(image,viewport):
    h,w=image.shape[:2]; hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV); s,v=hsv[:,:,1],hsv[:,:,2]
    neutral=((s<105)&(v>45)&(v<235)).astype(np.uint8)*255
    roi=np.zeros((h,w),np.uint8); roi[viewport.top:viewport.bottom,viewport.left:viewport.right]=255
    for x1,y1,x2,y2 in viewport.excluded_rectangles: roi[y1:y2,x1:x2]=0
    neutral=cv2.bitwise_and(neutral,roi)
    horiz=cv2.morphologyEx(neutral,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(max(11,round(w*.013)),3)))
    vert=cv2.morphologyEx(neutral,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(3,max(11,round(h*.024)))))
    road=cv2.morphologyEx(cv2.bitwise_or(horiz,vert),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    count,labels,stats,_=cv2.connectedComponentsWithStats(road); clean=np.zeros_like(road); minimum=max(24,round(h*w*.00004))
    for label in range(1,count):
        if stats[label,cv2.CC_STAT_AREA]>=minimum: clean[labels==label]=255
    return clean

def _thin(mask):
    binary=(mask>0).astype(np.uint8)*255; skeleton=np.zeros_like(binary); element=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    for _ in range(100):
        eroded=cv2.erode(binary,element); opened=cv2.dilate(eroded,element)
        skeleton=cv2.bitwise_or(skeleton,cv2.subtract(binary,opened)); binary=eroded
        if cv2.countNonZero(binary)==0: break
    return skeleton

def _line_axes(mask,image_height,viewport):
    binary=mask>0; hp=np.sum(binary[:,viewport.left:viewport.right],axis=1); vp=np.sum(binary[viewport.top:viewport.bottom,:],axis=0)
    rows=_projection_peaks(hp,viewport.top,viewport.bottom,max(24.,float(np.max(hp))*.18),max(24,round(image_height*.045)),12)
    cols=_projection_peaks(vp,viewport.left,viewport.right,max(20.,float(np.max(vp))*.18),max(24,round(mask.shape[1]*.025)),18)
    return rows,cols

def _projection_peaks(profile,lower,upper,threshold,minimum_distance,limit):
    candidates=[i for i in range(lower,upper) if profile[i]>=threshold]; selected=[]
    for i in sorted(candidates,key=lambda item:(-profile[item],item)):
        if all(abs(i-p)>=minimum_distance for p in selected): selected.append(i)
        if len(selected)>=limit: break
    return sorted(selected)

def _skeleton_keypoints(skeleton,viewport,image_height):
    binary=(skeleton>0).astype(np.uint8); neighbors=cv2.filter2D(binary,cv2.CV_16S,np.ones((3,3),np.uint8))-binary
    points=[]; radius=max(6,round(image_height*.012))
    for selector,source in ((neighbors<=1,"road_endpoint"),(neighbors>=3,"road_junction")):
        mask=((binary==1)&selector).astype(np.uint8)*255
        mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
        count,_,stats,centroids=cv2.connectedComponentsWithStats(mask)
        for i in range(1,count):
            x,y=np.rint(centroids[i]).astype(int)
            if stats[i,cv2.CC_STAT_AREA]>=4 and viewport.contains(int(x),int(y)): points.append((int(x),int(y),source))
    corners=cv2.goodFeaturesToTrack(skeleton,80,.05,max(18,round(image_height*.03)),blockSize=7)
    if corners is not None:
        for x,y in np.rint(corners[:,0]).astype(int):
            if viewport.contains(int(x),int(y)): points.append((int(x),int(y),"road_turn"))
    return points

def road_connectivity(mask,x,y,image_height):
    h,w=mask.shape; near=max(5,round(image_height*.01)); far=max(22,round(image_height*.06)); half=max(3,round(image_height*.006))
    regions={"left":mask[max(0,y-half):min(h,y+half+1),max(0,x-far):max(0,x-near)],"right":mask[max(0,y-half):min(h,y+half+1),min(w,x+near):min(w,x+far)],"up":mask[max(0,y-far):max(0,y-near),max(0,x-half):min(w,x+half+1)],"down":mask[min(h,y+near):min(h,y+far),max(0,x-half):min(w,x+half+1)]}
    present=tuple(n for n,r in regions.items() if r.size and float(np.mean(r>0))>=.16)
    return min(1.,len(present)/3.),present

def dark_center_score(gray,x,y,image_height):
    h,w=gray.shape; outer=max(10,round(image_height*.022)); inner=max(4,round(image_height*.008))
    x1,x2,y1,y2=max(0,x-outer),min(w,x+outer+1),max(0,y-outer),min(h,y+outer+1); patch=gray[y1:y2,x1:x2]
    if not patch.size:return 0.
    yy,xx=np.ogrid[y1:y2,x1:x2]; distance=(xx-x)**2+(yy-y)**2
    center=patch[distance<=inner**2]; ring=patch[(distance>=(inner+2)**2)&(distance<=outer**2)]
    if not center.size or not ring.size:return 0.
    return float(np.clip((float(np.mean(ring)-np.mean(center))+10.)/70.,0,1))