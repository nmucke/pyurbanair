from pathlib import Path
import os
import sys

sys.path.insert(0, '/private/tmp/pyurbanair-sgs-pdf-deps')
os.environ['MPLCONFIGDIR'] = '/private/tmp/pyurbanair-mpl-pdf'
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
from matplotlib.textpath import TextPath
from matplotlib.path import Path as MPath
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/pdf/sgs_discrepancy_mathematical_note.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
W, H = 595.276, 841.89
LEFT, RIGHT = 51, 51
WIDTH = W - LEFT - RIGHT
INK = HexColor('#172535')
BLUE = HexColor('#205879')
GRAY = HexColor('#52616c')
BODY = ParagraphStyle('body', fontName='Times-Roman', fontSize=10.4,
                      leading=13.4, textColor=INK, spaceAfter=0)
SMALL = ParagraphStyle('small', parent=BODY, fontSize=8.4, leading=10.5)
HEAD = ParagraphStyle('head', fontName='Helvetica-Bold', fontSize=11,
                      leading=14, textColor=BLUE)
c = canvas.Canvas(str(OUT), pagesize=(W,H))
c.setTitle('Strain- and rotation-dependent SGS discrepancy: mathematical review note')
c.setAuthor('pyurbanair research notes')
c.setSubject('Proposed low-dimensional eddy-viscosity correction for ensemble data assimilation')
y = 0

def para(text, style=BODY, gap=5):
    global y
    p = Paragraph(text, style)
    _, height = p.wrap(WIDTH, H)
    y -= height
    p.drawOn(c, LEFT, y)
    y -= gap

def heading(text):
    global y
    y -= 7
    para(text, HEAD, 5)

def eq(math, number, size=12.0, gap=7):
    global y
    path = TextPath((0, 0), '$' + math + '$', size=size, usetex=False)
    box = path.get_extents()
    factor = min(1, (WIDTH-36)/box.width)
    height = box.height*factor
    y -= height + 6
    c.saveState()
    c.translate(LEFT + (WIDTH-24-box.width*factor)/2 - box.x0*factor,
                y - box.y0*factor)
    c.scale(factor, factor)
    dest = c.beginPath()
    last = (0.,0.)
    for v, code in path.iter_segments(curves=True):
        if code == MPath.MOVETO:
            dest.moveTo(*v); last = v
        elif code == MPath.LINETO:
            dest.lineTo(*v); last = v
        elif code == MPath.CURVE3:
            x0,y0 = last; x1,y1,x2,y2 = v
            dest.curveTo(x0+2*(x1-x0)/3, y0+2*(y1-y0)/3,
                         x2+2*(x1-x2)/3, y2+2*(y1-y2)/3, x2,y2)
            last = (x2,y2)
        elif code == MPath.CURVE4:
            dest.curveTo(*v); last=v[-2:]
        elif code == MPath.CLOSEPOLY:
            dest.close()
    c.setFillColor(INK)
    c.drawPath(dest, fill=1, stroke=0)
    c.restoreState()
    c.setFont('Times-Roman',10)
    c.setFillColor(GRAY)
    c.drawRightString(W-RIGHT, y+height/2-3, f'({number})')
    y -= gap

def begin(page):
    global y
    c.setFillColor(BLUE)
    c.rect(LEFT,H-40,WIDTH,2,fill=1,stroke=0)
    c.setFont('Helvetica',8)
    c.drawString(LEFT,H-31,'PYURBANAIR  /  MATHEMATICAL REVIEW NOTE')
    c.drawRightString(W-RIGHT,H-31,'29 SEPTEMBER 2026')
    y = H-63
    if page == 1:
        title = Paragraph('Strain- and rotation-dependent<br/>SGS discrepancy',
                          ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=22,
                                         leading=25,textColor=INK))
        _,h=title.wrap(WIDTH,H); y-=h; title.drawOn(c,LEFT,y); y-=10
        para('<b>Purpose.</b> Review a low-dimensional correction to an existing LES '
             'eddy-viscosity closure for urban-flow data assimilation. This is a proposed '
             'model, not a validated result. The first experiment estimates three '
             'coefficients; a fourth is an optional extension.')
    else:
        para('Physical properties and inference',
             ParagraphStyle('title2',fontName='Helvetica-Bold',fontSize=19,
                            leading=23,textColor=INK),8)

def end(page):
    assert y > 63, f'Page {page} overlaps footer: y={y}'
    print(f'Page {page}: last content baseline {y:.1f} pt')
    c.setStrokeColor(HexColor('#d1dbe2')); c.line(LEFT,48,W-RIGHT,48)
    c.setFont('Helvetica',8); c.setFillColor(GRAY)
    c.drawString(LEFT,34,'Proposed formulation | For colleague review')
    c.drawRightString(W-RIGHT,34,f'{page} / 2')
    c.showPage()

begin(1)
heading('1. Governing equation and baseline closure')
para('Assume incompressible, constant-density flow. Velocities are resolved LES '
     'quantities; repeated indices are summed. With kinematic pressure p* absorbing '
     'the isotropic SGS stress, write')
eq(r'\partial_t u_i+u_j\partial_j u_i=-\partial_i p^*'
   r'+\partial_j[2(\nu+\nu_t^{b})S_{ij}]+f_i(\theta),\qquad \partial_i u_i=0',1)
para('Here the molecular viscosity is unchanged; f contains the physical forcing '
     'with parameters <font face="Symbol">&#952;</font>. All viscosities have units '
     'm<super>2</super>/s. The baseline SGS viscosity '
     'is nonnegative and is evaluated from each member\'s current flow and grid, '
     'using the native closure (e.g. Vreman [1]).')
heading('2. Dimensionless local features')
eq(r'G_{ij}=\partial_j u_i,\qquad S=\frac{G+G^{\mathsf{T}}}{2},'
   r'\qquad \Omega=\frac{G-G^{\mathsf{T}}}{2}',2)
eq(r'q=\frac{\Omega:\Omega-S:S}{\Omega:\Omega+S:S+\epsilon_g^2},'
   r'\qquad s=\frac{T_{\rm ref}^2 S:S}{1+T_{\rm ref}^2 S:S}',3)
para('The colon denotes the Frobenius inner product. Thus -1 &lt; q &lt; 1 '
     'distinguishes strain-dominated (negative) from rotation-dominated (positive) '
     'flow; simple shear has q = 0. The bounded feature 0 &le; s &lt; 1 measures '
     'strain strength. Fix T<sub>ref</sub> = H/U<sub>ref</sub> and '
     '<font face="Symbol">&#949;</font><sub>g</sub> &gt; 0 [s<super>-1</super>] '
     'before inference. Both features vanish at zero gradient.')
para('For height dependence, choose a fixed band z<sub>a</sub> &lt; z &lt; '
     'z<sub>b</sub> around canopy height H. Use the following feature inside '
     'the band and zero outside it:')
eq(r'\phi(z)=\sin^2\!\left(\pi\frac{z-z_a}{z_b-z_a}\right)',4)
heading('3. Bounded multiplicative correction')
eq(r'g=b_0+b_1\phi(z)+b_2q+b_3s,\qquad'
   r'\nu_t^{b}=\nu_t^{0}\exp\!\left[L\tanh(g/L)\right],\quad L>0',5)
para('All b coefficients are dimensionless. They control overall mixing (b<sub>0</sub>), '
     'canopy-local mixing (b<sub>1</sub>), strain/rotation contrast (b<sub>2</sub>) '
     'and strain-strength response (b<sub>3</sub>). Set b<sub>3</sub> = 0 initially. '
     'The fixed log-amplitude cap L bounds the multiplier between exp(-L) and exp(L); '
     'near g = 0 it behaves as exp(g). Zero coefficients recover the native closure.')
end(1)

begin(2)
heading('4. Momentum and energy implications')
para('Relative to evaluating the baseline closure on the same instantaneous state, '
     'the added momentum tendency is')
eq(r'd_i=\partial_j\!\left[2(\nu_t^{b}-\nu_t^{0})S_{ij}\right]',6)
para('Evaluate this through the native conservative stress divergence, including '
     'spatial derivatives of the multiplier. Multiplying an already computed '
     'diffusion tendency is not equivalent. For the effective deviatoric stress,')
eq(r'\tau_{ij}^{b,\mathrm{dev}}=-2\nu_t^{b}S_{ij},\qquad'
   r'\Pi_b=-\tau_{ij}^{b,\mathrm{dev}}S_{ij}=2\nu_t^{b}(S:S)\geq0',7)
para('The continuum closure remains dissipative; the correction can reduce baseline '
     'dissipation but cannot produce net SGS backscatter. Stress alignment with strain '
     'is unchanged, and zero baseline viscosity remains zero. Integrated discrepancy '
     'equals a boundary stress flux, so zero net momentum contribution requires '
     'appropriate boundary fluxes; urban walls cannot be ignored. Discrete energy '
     'stability and wall treatment still require testing. The scalar features are '
     'invariant under fixed rotations of axes and uniform velocity translations, '
     'not necessarily under a change to a rotating reference frame.')
heading('5. Inference problem')
para('First hold physical forcing fixed and estimate b = (b<sub>0</sub>, '
     'b<sub>1</sub>, b<sub>2</sub>) over one window. For noisy observations y, let '
     'h(b) denote the observation operator applied to the full LES trajectory, '
     'including any prescribed temporal averaging. Use')
eq(r'y=h(b)+\varepsilon,\quad \varepsilon\sim\mathcal{N}(0,R),'
   r'\qquad b\sim\mathcal{N}(0,C_b)',8)
eq(r'J(b)=\frac{1}{2}[y-h(b)]^{\mathsf{T}}R^{-1}[y-h(b)]'
   r'+\frac{1}{2}b^{\mathsf{T}}C_b^{-1}b',9)
para('Assume independent noise and coefficients, with positive-definite covariances. '
     'R is the untempered covariance of the actual observation product; C<sub>b</sub> '
     'regularizes departure from the native closure. ESMDA approximates inference '
     'for this target; ensemble filtering may augment the state with b. Each member '
     'recomputes the features from its own evolving velocity. Hold the native global '
     'SGS constant fixed to avoid redundancy with b<sub>0</sub>. Only later estimate '
     'forcing jointly or introduce slowly evolving coefficients. Ensemble inference '
     'of closure parameters has precedent [3], but does not validate this LES model.')
heading('6. Questions for review and minimal validation')
para('<b>Physics:</b> Is a scalar viscosity multiplier adequate for roof-level '
     'momentum transport, or is an anisotropic stress correction needed? '
     '<b>Features:</b> Are q, the canopy band and the optional s sufficiently distinct '
     'across the sampled flow? Inspect the singular values of '
     'R<super>-1/2</super>(dh/db)C<sub>b</sub><super>1/2</super>. '
     '<b>Regularization:</b> Which prior scales and amplitude cap are defensible?')
para('Compare the native closure, a fitted global SGS constant, and the three-feature '
     'model at matched cost. Test held-out sensors and forecasts after assimilation '
     'stops; report mean profiles, Reynolds stresses, dissipation and uncertainty '
     'coverage. Freeze tuning before cross-solver/grid tests. Gradient-based SGS '
     'models [1,2] motivate the structure; transferability remains a hypothesis.')
heading('Selected references')
para('[1] Vreman (2004). An eddy-viscosity subgrid-scale model for turbulent shear '
     'flow. <i>Physics of Fluids</i> 16, 3670-3681. '
     '<link href="https://doi.org/10.1063/1.1785131" color="#205879">doi:10.1063/1.1785131</link>.<br/>'
     '[2] Nicoud &amp; Ducros (1999). Subgrid-scale stress modelling based on the '
     'square of the velocity gradient tensor. <i>Flow, Turbulence and Combustion</i> '
     '62, 183-200. <link href="https://doi.org/10.1023/A:1009995426001" color="#205879">doi:10.1023/A:1009995426001</link>.<br/>'
     '[3] Zhang et al. (2022). Ensemble Kalman method for learning turbulence models '
     'from indirect observation data. <i>Journal of Fluid Mechanics</i> 949, A26 '
     '(RANS study). <link href="https://doi.org/10.1017/jfm.2022.744" color="#205879">doi:10.1017/jfm.2022.744</link>.',SMALL,0)
end(2)
c.save()
reader=PdfReader(OUT)
assert len(reader.pages)==2
for p in reader.pages:
    assert len(p.extract_text())>1000
print(OUT)
