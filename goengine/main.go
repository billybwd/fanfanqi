package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	rows = 8
	cols = 4
	nsq = rows * cols
	empty = 0
	hidden = 15
	red = 0
	black = 1
	king = 0
	cannon = 5
	soldier = 6
	moveKind = 1
	captureKind = 2
	hiddenCaptureKind = 3
	flipKind = 4
	defaultLongChaseLimit = 5
	defaultNoCaptureLimit = 16
)

var (
	pieceTypes = []string{"将", "仕", "象", "车", "马", "炮", "兵"}
	counts = [7]int{1, 2, 2, 2, 2, 2, 5}
	ranks = [7]int{6, 5, 4, 3, 2, 1, 0}
	hpValue = [7]int{30, 10, 5, 5, 5, 5, 2}
	flipValue = [7]float64{12, 8, 8, 4, 4, 6, 2}
	sideOf = [16]int{-1}
	typeOf = [16]int{-1}
	hpc = [16]int{}
	flipCost = [16]float64{}
	canEat = [16][16]bool{}
	neighbors [nsq][]int
	rays [nsq][4][]int
	distance [nsq][nsq]int
	zSquare [nsq][16]uint64 // zSquare[sq][empty] == 0
	zPool [15][8]uint64
)

func splitmix(state *uint64) uint64 {
	*state += 0x9E3779B97F4A7C15
	z := *state
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

func init() {
	seed := uint64(20261028)
	for sq := 0; sq < nsq; sq++ { for code := 1; code < 16; code++ { zSquare[sq][code] = splitmix(&seed) } }
	for code := 1; code < 15; code++ { for n := range zPool[code] { zPool[code][n] = splitmix(&seed) } }
}

func (p *Position) computeHash() uint64 {
	h := uint64(0)
	for sq, v := range p.B { h ^= zSquare[sq][int(v)] }
	for code := 1; code < 15; code++ { h ^= zPool[code][p.Pool[code]] }
	return h
}
func (p *Position) setSquare(sq int, code int8) { p.Hash ^= zSquare[sq][int(p.B[sq])] ^ zSquare[sq][int(code)]; p.B[sq] = code }
func (p *Position) takeFromPool(code int) { p.Hash ^= zPool[code][p.Pool[code]] ^ zPool[code][p.Pool[code]-1]; p.Pool[code]-- }

func init() {
	for s := 0; s < 2; s++ {
		for t := 0; t < 7; t++ {
			code := s*7 + t + 1
			sideOf[code], typeOf[code] = s, t
			hpc[code], flipCost[code] = hpValue[t], flipValue[t]
		}
	}
	for a := 1; a <= 14; a++ {
		for d := 1; d <= 14; d++ {
			if sideOf[a] == sideOf[d] { continue }
			ta, td := typeOf[a], typeOf[d]
			if ta == soldier && td == king { canEat[a][d] = true
			} else if ta == king && td == soldier { canEat[a][d] = false
			} else { canEat[a][d] = ranks[ta] >= ranks[td] }
		}
	}
	dirs := [4][2]int{{-1,0},{1,0},{0,-1},{0,1}}
	for sq := 0; sq < nsq; sq++ {
		r, c := sq/cols, sq%cols
		for di, d := range dirs {
			nr, nc := r+d[0], c+d[1]
			if nr >= 0 && nr < rows && nc >= 0 && nc < cols { neighbors[sq] = append(neighbors[sq], nr*cols+nc) }
			for nr >= 0 && nr < rows && nc >= 0 && nc < cols {
				rays[sq][di] = append(rays[sq][di], nr*cols+nc)
				nr, nc = nr+d[0], nc+d[1]
			}
		}
		for other := 0; other < nsq; other++ {
			distance[sq][other] = abs(sq/cols-other/cols)+abs(sq%cols-other%cols)
		}
	}
}

type Move struct { Kind, A, C int }
type Position struct {
	B [nsq]int8
	Pool [15]int
	HP [2]float64
	Turn int
	Cur, Max [2]int
	ComboScore [2]int
	Flip float64
	Winner int
	NoCapturePlies int
	ChaseCount [2]int
	ChaseTarget [2]int
	LongChaseLimit int
	NoCaptureLimit int
	Hash uint64 // Zobrist hash of B and Pool, updated incrementally
}
type Undo struct {
	Hash uint64
	A, C int
	PA, PC int8
	HP [2]float64
	Turn int
	Cur, Max [2]int
	ComboScore [2]int
	Flip float64
	Winner int
	NoCapturePlies int
	ChaseCount [2]int
	ChaseTarget [2]int
	PoolCode int
}
type State struct {
	Turn string `json:"turn"`
	HP map[string]float64 `json:"hp"`
	Board [][]string `json:"board"`
	Captured map[string]map[string]int `json:"captured"`
	Combo map[string]int `json:"combo"`
	MaxCombo map[string]int `json:"max_combo"`
	ComboScore map[string]int `json:"combo_score"`
	FlipScore float64 `json:"flip_score"`
	NoCapturePlies int `json:"no_capture_plies"`
	ChaseCount map[string]int `json:"chase_count"`
	ChaseTarget map[string]*int `json:"chase_target"`
	Rules map[string]int `json:"rules"`
}
type Request struct {
	Strategy string `json:"strategy"`
	Threads int `json:"threads"`
	TimeBudgetMS int `json:"time_budget_ms"`
	State State `json:"state"`
	Params map[string]any `json:"params"`
	Progress bool `json:"progress"` // also emit one {"progress":...} line per completed depth
}
type Response struct {
	Action map[string]any `json:"action"`
	Info map[string]any `json:"info"`
	Error string `json:"error,omitempty"`
}

func codeFor(side, typ int) int { return side*7 + typ + 1 }
func sideKey(s int) string { if s == red { return "red" }; return "black" }
func abs(v int) int { if v < 0 { return -v }; return v }

func parsePosition(s State) (Position, error) {
	var p Position
	p.Winner = -1
	p.ChaseTarget = [2]int{-1, -1}
	p.LongChaseLimit = defaultLongChaseLimit
	p.NoCaptureLimit = defaultNoCaptureLimit
	if s.Turn == "black" { p.Turn = black } else if s.Turn != "red" { return p, fmt.Errorf("turn must be red or black") }
	if len(s.Board) != rows { return p, fmt.Errorf("board must have %d rows", rows) }
	seen := [2][7]int{}
	for r := 0; r < rows; r++ {
		if len(s.Board[r]) != cols { return p, fmt.Errorf("board row %d must have %d cells", r, cols) }
		for c, token := range s.Board[r] {
			sq := r*cols+c
			switch {
			case token == "." || token == "" || token == "空":
				p.B[sq] = empty
			case token == "?" || token == "暗":
				p.B[sq] = hidden
			case len([]rune(token)) == 2 && (strings.HasPrefix(token, "r") || strings.HasPrefix(token, "b")):
				runes := []rune(token)
				side := red
				if runes[0] == 'b' { side = black }
				typ := -1
				for i, name := range pieceTypes { if runes[1] == []rune(name)[0] { typ = i; break } }
				if typ < 0 { return p, fmt.Errorf("unknown piece %q", token) }
				p.B[sq] = int8(codeFor(side, typ))
				seen[side][typ]++
			default:
				return p, fmt.Errorf("invalid board token %q", token)
			}
		}
	}
	for side := 0; side < 2; side++ {
		key := sideKey(side)
		if hp, ok := s.HP[key]; ok { p.HP[side] = hp } else { p.HP[side] = 60 }
		if s.Combo != nil { p.Cur[side] = s.Combo[key] }
		if s.MaxCombo != nil { p.Max[side] = s.MaxCombo[key] }
		if s.ComboScore != nil { p.ComboScore[side] = s.ComboScore[key] } else if p.Max[side] >= 3 && (p.Cur[side] < 3 || p.Max[side] > p.Cur[side]) { p.ComboScore[side] = p.Max[side] }
		for typ, name := range pieceTypes {
			captured := 0
			if s.Captured != nil && s.Captured[key] != nil { captured = s.Captured[key][name] }
			remaining := counts[typ] - captured - seen[side][typ]
			if remaining < 0 { return p, fmt.Errorf("piece count inconsistent for %s %s", key, name) }
			p.Pool[codeFor(side, typ)] = remaining
		}
	}
	p.Flip = s.FlipScore
	p.NoCapturePlies = s.NoCapturePlies
	if limit := s.Rules["long_chase_limit"]; limit >= 2 { p.LongChaseLimit = limit }
	if limit := s.Rules["no_capture_limit"]; limit >= 1 { p.NoCaptureLimit = limit }
	for side := 0; side < 2; side++ {
		key := sideKey(side)
		p.ChaseCount[side] = s.ChaseCount[key]
		if target, ok := s.ChaseTarget[key]; ok && target != nil && *target >= 0 && *target < nsq {
			p.ChaseTarget[side] = *target
		}
	}
	p.Hash = p.computeHash()
	return p, nil
}

func (p *Position) gen() (caps, hiddenCaps, moves, flips []Move) { return p.genInto(&plyBuf{}) }

// plyBuf holds reusable move lists for one search ply, so the hot path does not allocate.
type plyBuf struct {
	caps, hcaps, moves, flips, legal, ordered, tactical, quietMoves []Move
	scored, quiet, flipScores []scoredMove
}

func (p *Position) genInto(buf *plyBuf) (caps, hiddenCaps, moves, flips []Move) {
	caps, hiddenCaps, moves, flips = buf.caps[:0], buf.hcaps[:0], buf.moves[:0], buf.flips[:0]
	for sq, raw := range p.B {
		piece := int(raw)
		if piece == empty { continue }
		if piece == hidden { flips = append(flips, Move{flipKind,sq,sq}); continue }
		if sideOf[piece] != p.Turn { continue }
		if typeOf[piece] == cannon {
			for _, ray := range rays[sq] {
				mounted := false
				for _, target := range ray {
					q := int(p.B[target]); if q == empty { continue }
					if !mounted { mounted = true; continue }
					if q == hidden { hiddenCaps = append(hiddenCaps, Move{hiddenCaptureKind,sq,target})
					} else if sideOf[q] != p.Turn { caps = append(caps, Move{captureKind,sq,target}) }
					break
				}
			}
			continue
		}
		for _, target := range neighbors[sq] {
			q := int(p.B[target])
			if q == empty { moves = append(moves, Move{moveKind,sq,target})
			} else if q != hidden && canEat[piece][q] { caps = append(caps, Move{captureKind,sq,target}) }
		}
	}
	legalMoves := buf.legal[:0]
	for _, m := range moves {
		u := p.doMove(m.A, m.C)
		violates := p.ChaseCount[p.Turn^1] >= p.LongChaseLimit-1
		p.undo(u)
		if !violates { legalMoves = append(legalMoves, m) }
	}
	buf.caps, buf.hcaps, buf.moves, buf.flips, buf.legal = caps, hiddenCaps, moves, flips, legalMoves
	moves = legalMoves
	return
}

// capturesInto lists captures of face-up pieces in the same order as gen().
func (p *Position) capturesInto(caps []Move) []Move {
	caps = caps[:0]
	for sq, raw := range p.B {
		piece := int(raw)
		if piece == empty || piece == hidden || sideOf[piece] != p.Turn { continue }
		if typeOf[piece] == cannon {
			for _, ray := range rays[sq] {
				mounted := false
				for _, target := range ray {
					q := int(p.B[target]); if q == empty { continue }
					if !mounted { mounted = true; continue }
					if q != hidden && sideOf[q] != p.Turn { caps = append(caps, Move{captureKind,sq,target}) }
					break
				}
			}
			continue
		}
		for _, target := range neighbors[sq] { q := int(p.B[target]); if q != empty && q != hidden && canEat[piece][q] { caps = append(caps, Move{captureKind,sq,target}) } }
	}
	return caps
}
func (p *Position) save(a,c,poolCode int) Undo {
	return Undo{Hash:p.Hash,A:a,C:c,PA:p.B[a],PC:p.B[c],HP:p.HP,Turn:p.Turn,Cur:p.Cur,Max:p.Max,ComboScore:p.ComboScore,Flip:p.Flip,Winner:p.Winner,NoCapturePlies:p.NoCapturePlies,ChaseCount:p.ChaseCount,ChaseTarget:p.ChaseTarget,PoolCode:poolCode}
}
func (p *Position) undo(u Undo) {
	p.B[u.A], p.B[u.C] = u.PA, u.PC
	p.Hash = u.Hash
	p.HP, p.Turn, p.Cur, p.Max, p.Flip, p.Winner = u.HP, u.Turn, u.Cur, u.Max, u.Flip, u.Winner
	p.ComboScore = u.ComboScore
	p.NoCapturePlies, p.ChaseCount, p.ChaseTarget = u.NoCapturePlies, u.ChaseCount, u.ChaseTarget
	if u.PoolCode > 0 { p.Pool[u.PoolCode]++ }
}
func (p *Position) hit(side int) { p.Cur[side]++; if p.Cur[side] > p.Max[side] { p.Max[side] = p.Cur[side] } }
func (p *Position) breakCombo(side int) { if p.Cur[side] >= 3 { p.ComboScore[side] += p.Cur[side] }; p.Cur[side] = 0 }
func (p *Position) checkOver() {
	if p.HP[red] <= 0 { p.Winner = black } else if p.HP[black] <= 0 { p.Winner = red }
}
func (p *Position) doMove(a,c int) Undo {
	u := p.save(a,c,0); attacker, target := int(p.B[a]), int(p.B[c]); p.setSquare(c, p.B[a]); p.setSquare(a, empty)
	if target != empty { p.HP[sideOf[target]] -= float64(hpc[target]); if sideOf[target] != p.Turn { p.hit(p.Turn) } else { p.breakCombo(p.Turn) }; p.checkOver()
	} else { p.breakCombo(p.Turn) }
	me, opponent := p.Turn, p.Turn^1
	if target == empty { p.NoCapturePlies++ } else { p.NoCapturePlies = 0 }
	if target != empty {
		for side := 0; side < 2; side++ { if p.ChaseTarget[side] == c { p.ChaseTarget[side], p.ChaseCount[side] = -1, 0 } }
	}
	if p.ChaseTarget[opponent] == a { p.ChaseTarget[opponent] = c }
	if target != empty { p.ChaseTarget[me], p.ChaseCount[me] = -1, 0
	} else {
		first, found := p.captureTargetScan(c, me, p.ChaseTarget[me])
		if found { p.ChaseCount[me]++ } else if first >= 0 { p.ChaseTarget[me], p.ChaseCount[me] = first, 1 } else { p.ChaseTarget[me], p.ChaseCount[me] = -1, 0 }
	}
	p.Turn ^= 1
	_ = attacker
	return u
}
// captureTargetScan is captureTargetsFrom without allocating: the first target (or -1) and whether old is a target.
func (p *Position) captureTargetScan(sq, side, old int) (first int, found bool) {
	first = -1
	code := int(p.B[sq]); if code == empty || code == hidden || sideOf[code] != side { return }
	note := func(target int) { if first < 0 { first = target }; if target == old { found = true } }
	if typeOf[code] == cannon {
		for _, ray := range rays[sq] { mounted := false; for _, target := range ray { q := int(p.B[target]); if q == empty { continue }; if !mounted { mounted = true; continue }; if q != hidden && sideOf[q] != side { note(target) }; break } }
		return
	}
	for _, target := range neighbors[sq] { q:=int(p.B[target]); if q!=empty && q!=hidden && sideOf[q]!=side && canEat[code][q] { note(target) } }
	return
}
func (p *Position) captureTargetsFrom(sq, side int) []int {
	code := int(p.B[sq]); if code == empty || code == hidden || sideOf[code] != side { return nil }
	targets := []int{}
	if typeOf[code] == cannon {
		for _, ray := range rays[sq] { mounted := false; for _, target := range ray { q := int(p.B[target]); if q == empty { continue }; if !mounted { mounted = true; continue }; if q != hidden && sideOf[q] != side { targets = append(targets,target) }; break } }
		return targets
	}
	for _, target := range neighbors[sq] { q:=int(p.B[target]); if q!=empty && q!=hidden && sideOf[q]!=side && canEat[code][q] { targets=append(targets,target) } }
	return targets
}
func (p *Position) doFlip(sq, code int) Undo {
	u := p.save(sq,sq,code); p.setSquare(sq, int8(code)); p.takeFromPool(code); p.ChaseCount[p.Turn],p.ChaseTarget[p.Turn]=0,-1; p.breakCombo(p.Turn); p.Turn ^= 1; return u
}
func (p *Position) doFlipApprox(sq int) Undo {
	u := p.save(sq,sq,0); p.ChaseCount[p.Turn],p.ChaseTarget[p.Turn]=0,-1; p.breakCombo(p.Turn); p.Turn ^= 1; return u
}
func (p *Position) doHiddenCapture(a,c,code int) Undo {
	u := p.save(a,c,code); p.takeFromPool(code); p.HP[sideOf[code]] -= float64(hpc[code]); p.setSquare(c, p.B[a]); p.setSquare(a, empty)
	me,opponent:=p.Turn,p.Turn^1; if sideOf[code] != me { p.hit(me) } else { p.breakCombo(me) }; p.checkOver(); p.NoCapturePlies=0; if p.ChaseTarget[opponent]==a {p.ChaseTarget[opponent]=c}; p.ChaseTarget[me],p.ChaseCount[me]=-1,0; p.Turn ^= 1; return u
}
func (p *Position) attacked(sq, piece int) bool {
	enemy := sideOf[piece]^1
	for _, n := range neighbors[sq] { q:=int(p.B[n]); if q!=empty && q!=hidden && sideOf[q]==enemy && typeOf[q]!=cannon && canEat[q][piece] { return true } }
	for _, ray := range rays[sq] { mounted:=false; for _, n:=range ray { q:=int(p.B[n]); if q==empty {continue}; if !mounted {mounted=true; continue}; if q!=hidden && sideOf[q]==enemy && typeOf[q]==cannon {return true}; break } }
	return false
}
func (p *Position) cannonHits(from,target int) bool {
	for _, ray := range rays[from] { mounted:=false; for _, n:=range ray { if p.B[n]==empty {continue}; if !mounted {mounted=true;continue}; if n==target {return true}; break } }
	return false
}
func (p *Position) comboFactor() int { total:=p.ComboScore[red]+p.ComboScore[black]; for side:=0;side<2;side++{if p.Cur[side]>=3{total+=p.Cur[side]}}; if total==0{return 1}; return total }
func (p *Position) hpDiff() float64 { return math.Abs(p.HP[red]-p.HP[black]) }
func (p *Position) multiplier() float64 { return float64(p.comboFactor())*(p.Flip+p.hpDiff()) }

type Weights struct {
	HP float64 `json:"hp"`
	PieceValue map[string]float64 `json:"piece_value"`
	Mobility float64 `json:"mobility"`
	Hanging float64 `json:"hanging"`
	HangingOne float64 `json:"hanging_one"`
	Pool float64 `json:"pool"`
	KingSoldierAdj float64 `json:"king_soldier_adj"`
	KingSoldierNear float64 `json:"king_soldier_near"`
	KingCannon float64 `json:"king_cannon"`
	KingTrapped float64 `json:"king_trapped"`
	Tempo float64 `json:"tempo"`
}
const maxPly = 128

type Searcher struct {
	P map[string]any
	W Weights
	PV [7]float64
	Deadline time.Time
	Nodes int64
	TT *TransTable
	Stop *atomic.Bool
	Strategy int
	Interval int64
	EvalCache map[uint64]float64
	EvalCacheSize int
	DeltaMargin float64
	CannonFull float64
	CannonEndgame float64
	// Parameters read once from P; map lookups are too slow for the hot path.
	Lam, Kappa, Temp float64
	MaxDepth, FullChancePlies, NodeQuietMoves, QDepth, QHiddenDepth, QHiddenCaps int
	RootFlips, NodeFlips, EarlyPool, MidPool, EarlyRootFlips, MidRootFlips, EarlyNodeFlips, MidNodeFlips, EarlyChance, MidChance int
	OnDepth func(best Move,value float64,depth int) // called after each completed iteration
	bufs [maxPly]plyBuf
	capBufs [maxPly][]Move
}
type TTEntry struct { Depth int; Value float64; Flag uint8; Best Move; HasBest bool }
type scoredMove struct { Score float64; M Move }
type timeoutPanic struct{}

// TransTable is a fixed-size transposition table shared by all search threads (lazy SMP).
// A slot stores key^value^meta, so a torn concurrent write fails the check and reads as a miss.
type ttSlot struct { check, value, meta atomic.Uint64 }
type TransTable struct { slots []ttSlot; mask uint64 }

func newTransTable(bits int) *TransTable {
	if bits < 10 { bits = 10 } else if bits > 26 { bits = 26 }
	return &TransTable{slots: make([]ttSlot, 1<<bits), mask: uint64(1<<bits - 1)}
}
func (t *TransTable) get(key uint64) (TTEntry, bool) {
	slot := &t.slots[key&t.mask]
	check, v, m := slot.check.Load(), slot.value.Load(), slot.meta.Load()
	if m == 0 || check^v^m != key { return TTEntry{}, false }
	e := TTEntry{Depth: int(m >> 1 & 0xff), Value: math.Float64frombits(v), Flag: uint8(m >> 9 & 3), HasBest: m>>11&1 == 1}
	e.Best = Move{Kind: int(m >> 12 & 7), A: int(m >> 15 & 63), C: int(m >> 21 & 63)}
	return e, true
}
func (t *TransTable) put(key uint64, e TTEntry) {
	m := uint64(1) | uint64(e.Depth&0xff)<<1 | uint64(e.Flag&3)<<9 | uint64(e.Best.Kind&7)<<12 | uint64(e.Best.A&63)<<15 | uint64(e.Best.C&63)<<21
	if e.HasBest { m |= 1 << 11 }
	v := math.Float64bits(e.Value)
	slot := &t.slots[key&t.mask]
	slot.value.Store(v); slot.meta.Store(m); slot.check.Store(key ^ v ^ m)
}

func number(v any, fallback float64) float64 { if n,ok:=v.(float64); ok{return n}; if n,ok:=v.(int);ok{return float64(n)};return fallback }
func intParam(p map[string]any,key string,fallback int) int { return int(number(p[key],float64(fallback))) }
func floatParam(p map[string]any,key string,fallback float64) float64 { return number(p[key],fallback) }
func weightMap(p map[string]any) map[string]any { if w,ok:=p["weights"].(map[string]any);ok{return w};return nil }
func defaultParams() map[string]any { return map[string]any{"multiplier_exponent":1.0,"loss_aversion":1.0,"win_temperature":12.0,"max_depth":8,"root_flips":8,"node_flips":3,"node_quiet_moves":10,"qsearch_depth":4,"full_chance_plies":1,"q_hidden_caps":1,"q_hidden_depth":0,"delta_margin":6.0,"eval_cache_size":200000,"cannon_endgame_value":2.0,"time_check_interval":32,"early_pool_threshold":20,"mid_pool_threshold":12,"early_root_flips":6,"mid_root_flips":7,"early_node_flips":2,"mid_node_flips":3,"early_chance_samples":6,"mid_chance_samples":8,"tt_bits":21} }
func makeSearcher(params map[string]any,strategy int,deadline time.Time,tt *TransTable,stop *atomic.Bool) *Searcher {
	d:=defaultParams(); for k,v:=range params {d[k]=v}; p:=d
	wm:=weightMap(p); nested:=map[string]any{}; if raw,ok:=wm["piece_value"].(map[string]any);ok{nested=raw}
	w:=Weights{HP:1,Mobility:0.3,Hanging:0.6,HangingOne:0.15,Pool:0.3,KingSoldierAdj:8,KingSoldierNear:3,KingCannon:5,KingTrapped:2,Tempo:0.5}
	if wm!=nil {w.HP=number(wm["hp"],w.HP);w.Mobility=number(wm["mobility"],w.Mobility);w.Hanging=number(wm["hanging"],w.Hanging);w.HangingOne=number(wm["hanging_one"],w.HangingOne);w.Pool=number(wm["pool"],w.Pool);w.KingSoldierAdj=number(wm["king_soldier_adj"],w.KingSoldierAdj);w.KingSoldierNear=number(wm["king_soldier_near"],w.KingSoldierNear);w.KingCannon=number(wm["king_cannon"],w.KingCannon);w.KingTrapped=number(wm["king_trapped"],w.KingTrapped);w.Tempo=number(wm["tempo"],w.Tempo)}
	pv:= [7]float64{10,6,3,2.5,2,5,1.5}; for i,t:=range pieceTypes {pv[i]=number(nested[t],pv[i])}
	interval:=intParam(p,"time_check_interval",32); if strategy==16 {interval=256}; if interval<1{interval=1}
	cacheSize,delta,cannonEndgame:=0,-1.0,-1.0; if strategy==17 {cacheSize=intParam(p,"eval_cache_size",200000);delta=floatParam(p,"delta_margin",6);cannonEndgame=floatParam(p,"cannon_endgame_value",2)}; if cacheSize<0{cacheSize=0}
	if tt==nil {tt=newTransTable(intParam(p,"tt_bits",21))}; if stop==nil {stop=&atomic.Bool{}}
	s:=&Searcher{P:p,W:w,PV:pv,Deadline:deadline,TT:tt,Stop:stop,Strategy:strategy,Interval:int64(interval),EvalCache:map[uint64]float64{},EvalCacheSize:cacheSize,DeltaMargin:delta,CannonFull:pv[cannon],CannonEndgame:cannonEndgame}
	s.Lam,s.Kappa,s.Temp=floatParam(p,"multiplier_exponent",1),floatParam(p,"loss_aversion",1),floatParam(p,"win_temperature",12)
	s.MaxDepth,s.FullChancePlies,s.NodeQuietMoves=intParam(p,"max_depth",8),intParam(p,"full_chance_plies",1),intParam(p,"node_quiet_moves",10)
	s.QDepth,s.QHiddenDepth,s.QHiddenCaps=intParam(p,"qsearch_depth",4),intParam(p,"q_hidden_depth",1),intParam(p,"q_hidden_caps",1)
	s.RootFlips,s.NodeFlips,s.EarlyPool,s.MidPool=intParam(p,"root_flips",8),intParam(p,"node_flips",3),intParam(p,"early_pool_threshold",20),intParam(p,"mid_pool_threshold",12)
	s.EarlyRootFlips,s.MidRootFlips,s.EarlyNodeFlips,s.MidNodeFlips=intParam(p,"early_root_flips",6),intParam(p,"mid_root_flips",7),intParam(p,"early_node_flips",2),intParam(p,"mid_node_flips",3)
	s.EarlyChance,s.MidChance=intParam(p,"early_chance_samples",6),intParam(p,"mid_chance_samples",8)
	return s
}
func (s *Searcher) buf(ply int) *plyBuf { if ply<maxPly {return &s.bufs[ply]}; return &plyBuf{} }

func (s *Searcher) tick() { s.Nodes++; if s.Nodes%s.Interval==0 && (time.Now().After(s.Deadline)||s.Stop.Load()) {panic(timeoutPanic{})} }
// staticEval caches staticEvalRaw (V17): the evaluation only depends on board, pool, HP and side to move.
func (s *Searcher) staticEval(p *Position) float64 {
	if s.EvalCacheSize==0 {s.setCannonValue(p);return s.staticEvalRaw(p)}
	h:=p.Hash;mix:=func(v uint64){h^=v;h*=1099511628211};for _,v:=range p.HP{mix(math.Float64bits(v))};mix(uint64(p.Turn))
	if v,ok:=s.EvalCache[h];ok{return v};if len(s.EvalCache)>=s.EvalCacheSize{s.EvalCache=map[uint64]float64{}};s.setCannonValue(p);v:=s.staticEvalRaw(p);s.EvalCache[h]=v;return v
}
// setCannonValue (V17): cannon value scales with board occupancy (screens and targets), from
// cannon_endgame_value on an empty board to piece_value 炮 on a full one. Negative keeps it fixed.
func (s *Searcher) setCannonValue(p *Position) {
	if s.CannonEndgame<0 {return};occupied:=0;for _,v:=range p.B{if v!=empty{occupied++}}
	s.PV[cannon]=s.CannonEndgame+(s.CannonFull-s.CannonEndgame)*float64(occupied)/float64(len(p.B))
}
func (s *Searcher) staticEvalRaw(p *Position) float64 {
	me,op:=p.Turn,p.Turn^1; var material [2]float64; var mobility [2]float64; kings:=[2]int{-1,-1}
	var soldiers,cannons [2][8]int; var nSoldiers,nCannons [2]int; threatSum,biggest,threatened:=0.0,0.0,false
	for sq,raw:=range p.B { code:=int(raw);if code==empty||code==hidden{continue}; side,t:=sideOf[code],typeOf[code];material[side]+=s.PV[t];if t==king{kings[side]=sq}else if t==soldier{soldiers[side][nSoldiers[side]]=sq;nSoldiers[side]++}else if t==cannon{cannons[side][nCannons[side]]=sq;nCannons[side]++};if t!=cannon {for _,n:=range neighbors[sq]{if p.B[n]==empty{mobility[side]++}}};if side==me&&p.attacked(sq,code){v:=float64(hpc[code])+s.PV[t];threatSum+=v;if v>biggest{biggest=v};threatened=true}}
	score:=s.W.HP*(p.HP[me]-p.HP[op])+material[me]-material[op]+s.W.Mobility*(mobility[me]-mobility[op])
	if threatened {score-=s.W.Hanging*(threatSum-biggest)+s.W.HangingOne*biggest}
	var poolValue [2]float64;for c:=1;c<=14;c++{if p.Pool[c]>0{poolValue[sideOf[c]]+=float64(p.Pool[c])*s.PV[typeOf[c]]}};score+=s.W.Pool*(poolValue[me]-poolValue[op])
	for side:=0;side<2;side++{k:=kings[side];if k<0{continue};enemy:=side^1;danger:=0.0;for _,sq:=range soldiers[enemy][:nSoldiers[enemy]]{d:=distance[k][sq];if d==1{danger+=s.W.KingSoldierAdj}else if d==2{danger+=s.W.KingSoldierNear}};for _,sq:=range cannons[enemy][:nCannons[enemy]]{if p.cannonHits(sq,k){danger+=s.W.KingCannon}};free:=0;for _,n:=range neighbors[k]{if p.B[n]==empty{free++}};if free<2{danger+=s.W.KingTrapped*float64(2-free)};if side==me{score-=danger}else{score+=danger}}
	return score+s.W.Tempo
}
func (s *Searcher) estMultiplier(p *Position) float64 {m:=float64(p.comboFactor())*(p.Flip+p.hpDiff());if m<1{return 1};return m}
func (s *Searcher) objective(v,m float64) float64 {switch s.Lam{case 1:v*=m;case 0:default:v*=math.Pow(m,s.Lam)};if v<0{v*=s.Kappa};return v}
func (s *Searcher) leafValue(p *Position) float64 {return s.valueFromEval(p,s.staticEval(p))}
func (s *Searcher) valueFromEval(p *Position,score float64) float64 {x:=score/s.Temp;if x>30{x=30}else if x< -30{x=-30};prob:=2/(1+math.Exp(-x))-1;return s.objective(prob,s.estMultiplier(p))}
func (s *Searcher) terminalValue(p *Position,ply int) float64 {m:=math.Max(1,p.multiplier());v:=(1+0.01*math.Log(m))*(1-0.001*float64(ply));if p.Winner!=p.Turn{v=-v};return s.objective(v,m)}

func (s *Searcher) quietScore(p *Position,a,c int) float64 {
	piece:=int(p.B[a]);score:=0.0;if p.attacked(a,piece){score+=float64(hpc[piece])*10+s.PV[typeOf[piece]]*10};u:=p.doMove(a,c);if p.attacked(c,piece){score-=(float64(hpc[piece])+s.PV[typeOf[piece]])*10};for _,n:=range neighbors[c]{q:=int(p.B[n]);if q!=empty&&q!=hidden&&canEat[piece][q]{score+=float64(hpc[q])*3}};p.undo(u);return score
}
func (s *Searcher) flipFull(p *Position,sq int,total int) float64 {
	me:=p.Turn; var own,enemy [4]int; nOwn,nEnemy:=0,0;for _,n:=range neighbors[sq]{q:=int(p.B[n]);if q!=empty&&q!=hidden{if sideOf[q]==me{own[nOwn]=q;nOwn++}else{enemy[nEnemy]=q;nEnemy++}}};score:=0.0
	for code:=1;code<=14;code++{if p.Pool[code]==0{continue};prob:=float64(p.Pool[code])/float64(total);t:=typeOf[code];gain:=0.0;if sideOf[code]==me{gain=s.PV[t];for _,e:=range enemy[:nEnemy]{if typeOf[e]!=cannon&&canEat[e][code]{gain-=float64(hpc[code])+s.PV[t];break}}}else{gain=-s.PV[t];if t!=cannon{for _,o:=range own[:nOwn]{if canEat[code][o]{gain-=float64(hpc[o])+s.PV[typeOf[o]];break}}}};score+=prob*gain};return score
}
func (s *Searcher) flipFast(p *Position,sq int) float64 {score:=0.0;for _,n:=range neighbors[sq]{q:=int(p.B[n]);if q!=empty&&q!=hidden{if sideOf[q]==p.Turn{score++}else{score-=1+float64(ranks[typeOf[q]])}}};return score}
func (s *Searcher) phaseLimits(p *Position)(int,int){pool:=poolSize(p);if s.Strategy!=17{return s.RootFlips,s.NodeFlips};if pool>=s.EarlyPool{return s.EarlyRootFlips,s.EarlyNodeFlips};if pool>=s.MidPool{return s.MidRootFlips,s.MidNodeFlips};return s.RootFlips,s.NodeFlips}
// sortScored is a stable sort by descending score (same order as sort.SliceStable, without allocating).
func sortScored(a []scoredMove) { for i:=1;i<len(a);i++ { x:=a[i]; j:=i; for j>0 && a[j-1].Score < x.Score { a[j]=a[j-1]; j-- }; a[j]=x } }
func moveToFront(moves []Move,m Move) { for i,x:=range moves { if x==m { copy(moves[1:i+1],moves[:i]); moves[0]=m; return } } }
func (s *Searcher) orderedMoves(p *Position,ply int,ttBest Move,hasTT bool) []Move {
	buf:=s.buf(ply);caps,hcaps,moves,flips:=p.genInto(buf);scored:=buf.scored[:0];root:=ply==0
	for _,m:=range caps{v:=int(p.B[m.C]);att:=int(p.B[m.A]);scored=append(scored,scoredMove{10000+float64(hpc[v])*20+s.PV[typeOf[v]]*5-float64(ranks[typeOf[att]]),m})}
	if len(hcaps)>0{total:=poolSize(p);enemy:=p.Turn^1;expected:=0.0;for c:=1;c<=14;c++{if p.Pool[c]>0{sign:=-1.0;if sideOf[c]==enemy{sign=1};expected+=float64(p.Pool[c]*hpc[c])*sign}};if total>0{expected/=float64(total)};for _,m:=range hcaps{scored=append(scored,scoredMove{5000+expected*20,m})}}
	quiet:=buf.quiet[:0];for _,m:=range moves{quiet=append(quiet,scoredMove{s.quietScore(p,m.A,m.C),m})};sortScored(quiet);buf.quiet=quiet;if !root{limit:=max(0,s.NodeQuietMoves);if len(quiet)>limit{quiet=quiet[:limit]}};scored=append(scored,quiet...)
	rootLimit,nodeLimit:=s.phaseLimits(p);flipScores:=buf.flipScores[:0];if len(flips)>0{total:=poolSize(p);for _,m:=range flips{v:=s.flipFast(p,m.A);if root&&total>0{v=s.flipFull(p,m.A,total)};if root{v-=1000};flipScores=append(flipScores,scoredMove{v,m})};sortScored(flipScores);buf.flipScores=flipScores;limit:=nodeLimit;if root{limit=rootLimit};if s.Strategy==17&&ply>=s.FullChancePlies{limit=min(limit,1)};if limit<len(flipScores){flipScores=flipScores[:max(0,limit)]}}
	sortScored(scored);buf.scored=scored;ordered:=buf.ordered[:0]
	if s.Strategy==17{
		tactical,quietMoves:=buf.tactical[:0],buf.quietMoves[:0];for _,x:=range scored{if x.M.Kind==captureKind||x.M.Kind==hiddenCaptureKind{tactical=append(tactical,x.M)}else{quietMoves=append(quietMoves,x.M)}};buf.tactical,buf.quietMoves=tactical,quietMoves
		ordered=append(ordered,tactical...);for i:=0;i<max(len(flipScores),len(quietMoves));i++{if i<len(flipScores){ordered=append(ordered,flipScores[i].M)};if i<len(quietMoves){ordered=append(ordered,quietMoves[i])}}
	} else {for _,x:=range scored{ordered=append(ordered,x.M)};for _,x:=range flipScores{ordered=append(ordered,x.M)}}
	if hasTT{moveToFront(ordered,ttBest)};buf.ordered=ordered;return ordered
}

// key extends the incremental board/pool hash with the remaining state that affects search results.
func (p *Position) key() uint64 {
	h:=p.Hash^1469598103934665603;mix:=func(v uint64){h^=v;h*=1099511628211};for _,v:=range p.HP{mix(math.Float64bits(v))};mix(uint64(p.Turn));for _,v:=range p.Cur{mix(uint64(v))};for _,v:=range p.Max{mix(uint64(v))};for _,v:=range p.ComboScore{mix(uint64(v))};mix(math.Float64bits(p.Flip));mix(uint64(p.Winner+1));mix(uint64(p.NoCapturePlies));for _,v:=range p.ChaseCount{mix(uint64(v))};for _,v:=range p.ChaseTarget{mix(uint64(v+1))};mix(uint64(p.LongChaseLimit));mix(uint64(p.NoCaptureLimit));return h
}

// sampleOffset is the systematic-sampling offset in [0,1) for a position: FNV-1a over board, pool and
// flip score, then splitmix64. Must match _sample_offset in fanfanqi/v17.py so both engines agree.
func (s *Searcher) sampleOffset(p *Position) float64 {
	h:=uint64(14695981039346656037);write:=func(b byte){h^=uint64(b);h*=1099511628211};for _,v:=range p.B{write(byte(v))};for _,v:=range p.Pool{write(byte(v))};bits:=math.Float64bits(p.Flip);for i:=0;i<8;i++{write(byte(bits>>(8*i)))}
	return float64(splitmix(&h)>>11)/float64(uint64(1)<<53)
}
// chanceWeights returns the probability of each pool code (0 = not an outcome), iterated in code order.
func (s *Searcher) chanceWeights(p *Position) (w [15]float64) {
	total:=poolSize(p);if total==0{return}
	n:=0;if s.Strategy==17{if total>=s.EarlyPool{n=s.EarlyChance}else if total>=s.MidPool{n=s.MidChance}}
	if n<=0 {for c:=1;c<=14;c++{if p.Pool[c]>0{w[c]=float64(p.Pool[c])/float64(total)}};return}
	protected:=func(c int)bool{return typeOf[c]==king||typeOf[c]==soldier};remainingSize:=0
	for c:=1;c<=14;c++{if p.Pool[c]==0{continue};if protected(c){w[c]=float64(p.Pool[c])/float64(total)}else{remainingSize+=p.Pool[c]}}
	if remainingSize==0{return};if remainingSize<=n{for c:=1;c<=14;c++{if p.Pool[c]>0&&!protected(c){w[c]=float64(p.Pool[c])/float64(total)}};return}
	offset:=s.sampleOffset(p);var sampled [15]int
	for i:=0;i<n;i++{x:=(float64(i)+offset)*float64(remainingSize)/float64(n);cum:=0;for c:=1;c<=14;c++{if p.Pool[c]==0||protected(c){continue};cum+=p.Pool[c];if x<float64(cum){sampled[c]++;break}}}
	mass:=float64(remainingSize)/float64(total);for c:=1;c<=14;c++{if sampled[c]>0{w[c]+=mass*float64(sampled[c])/float64(n)}};return
}
func (s *Searcher) child(p *Position,m Move,depth int,alpha,beta float64,ply int) float64 {
	if m.Kind==moveKind||m.Kind==captureKind{u:=p.doMove(m.A,m.C);v:=-s.negamax(p,depth-1,-beta,-alpha,ply+1);p.undo(u);return v}
	if m.Kind==flipKind&&ply>=s.FullChancePlies{u:=p.doFlipApprox(m.A);v:=-s.negamax(p,depth-1,-beta,-alpha,ply+1);p.undo(u);return v}
	weights:=s.chanceWeights(p);value:=0.0;for code:=1;code<=14;code++{prob:=weights[code];if prob<=0{continue};var u Undo;if m.Kind==flipKind{u=p.doFlip(m.A,code)}else{u=p.doHiddenCapture(m.A,m.C,code)};v:=-s.negamax(p,depth-1,math.Inf(-1),math.Inf(1),ply+1);p.undo(u);value+=prob*v};return value
}
func poolSize(p *Position)int{n:=0;for _,v:=range p.Pool{n+=v};return n}

func (s *Searcher) negamax(p *Position,depth int,alpha,beta float64,ply int) float64 {
	s.tick();if p.NoCapturePlies>=p.NoCaptureLimit{return 0};if p.Winner>=0{return s.terminalValue(p,ply)};if depth<=0{return s.qsearch(p,alpha,beta,ply,s.QDepth,s.QHiddenDepth)}
	key:=p.key();entry,ok:=s.TT.get(key);ttBest,hasTT:=entry.Best,ok&&entry.HasBest;alpha0:=alpha;if ok&&entry.Depth>=depth{if entry.Flag==0{return entry.Value};if entry.Flag==1&&entry.Value>=beta{return entry.Value};if entry.Flag==2&&entry.Value<=alpha{return entry.Value}}
	moves:=s.orderedMoves(p,ply,ttBest,hasTT);if len(moves)==0{return s.leafValue(p)};best:=-math.MaxFloat64;var bestMove Move;hasBest:=false
	for _,m:=range moves{v:=s.child(p,m,depth,alpha,beta,ply);if v>best{best,bestMove,hasBest=v,m,true};if v>alpha{alpha=v};if alpha>=beta{break}}
	flag:=uint8(0);if best<=alpha0{flag=2}else if best>=beta{flag=1};s.TT.put(key,TTEntry{Depth:depth,Value:best,Flag:flag,Best:bestMove,HasBest:hasBest});return best
}
func (s *Searcher) qsearch(p *Position,alpha,beta float64,ply,qdepth,hiddenLeft int)float64{
	s.tick();if p.NoCapturePlies>=p.NoCaptureLimit{return 0};if p.Winner>=0{return s.terminalValue(p,ply)};score:=s.staticEval(p);stand:=s.valueFromEval(p,score);if qdepth<=0||stand>=beta{return stand};if stand>alpha{alpha=stand}
	var caps []Move;if ply<maxPly{caps=p.capturesInto(s.capBufs[ply]);s.capBufs[ply]=caps}else{caps=p.capturesInto(nil)}
	// Biggest victim first (stable), as in the Python searcher.
	capKey:=func(m Move)int{return hpc[int(p.B[m.C])]*20-ranks[typeOf[int(p.B[m.A])]]};for i:=1;i<len(caps);i++{x:=caps[i];k:=capKey(x);j:=i;for j>0&&capKey(caps[j-1])<k{caps[j]=caps[j-1];j--};caps[j]=x}
	// Delta pruning (V17): skip a capture whose optimistic evaluation (captured HP + piece value + margin,
	// post-capture multiplier) cannot raise alpha; game-ending captures are always searched.
	for _,m:=range caps{victim:=int(p.B[m.C]);gain:=float64(hpc[victim])+s.PV[typeOf[victim]]+s.DeltaMargin;u:=p.doMove(m.A,m.C);if s.DeltaMargin>=0&&p.Winner<0&&s.valueFromEval(p,score+gain)<=alpha{p.undo(u);continue};v:=-s.qsearch(p,-beta,-alpha,ply+1,qdepth-1,hiddenLeft);p.undo(u);if v>=beta{return v};if v>alpha{alpha=v}}
	if s.Strategy==17&&hiddenLeft>0{_,hc,_,_:=p.gen();if len(hc)>0&&poolSize(p)>0{likely:=0;for c:=1;c<=14;c++{if p.Pool[c]>p.Pool[likely]{likely=c}};type candidate struct{m Move;score float64};cs:=[]candidate{};for _,m:=range hc{u:=p.doHiddenCapture(m.A,m.C,likely);score:=-s.leafValue(p);if p.Winner>=0{score=-s.terminalValue(p,ply+1)};p.undo(u);cs=append(cs,candidate{m,score})};sort.SliceStable(cs,func(i,j int)bool{return cs[i].score>cs[j].score});limit:=max(0,s.QHiddenCaps);if limit>len(cs){limit=len(cs)};for _,candidate:=range cs[:limit]{value:=0.0;weights:=s.chanceWeights(p);for code:=1;code<=14;code++{prob:=weights[code];if prob<=0{continue};u:=p.doHiddenCapture(candidate.m.A,candidate.m.C,code);v:=-s.qsearch(p,math.Inf(-1),math.Inf(1),ply+1,qdepth-1,hiddenLeft-1);p.undo(u);value+=prob*v};if value>=beta{return value};if value>alpha{alpha=value}}}}
	return alpha
}

func safeRootWindow(search *Searcher,p Position,m Move,depth int,alpha,beta float64)(value float64,ok bool){
	defer func(){if r:=recover();r!=nil{if _,yes:=r.(timeoutPanic);yes{ok=false}else{panic(r)}}}()
	value=search.child(&p,m,depth,alpha,beta,0);return value,true
}
func rootAction(m Move)map[string]any{if m.Kind==flipKind{return map[string]any{"type":"flip","row":m.A/cols,"col":m.A%cols}};kind:="move";if m.Kind==captureKind||m.Kind==hiddenCaptureKind{kind="capture"};return map[string]any{"type":kind,"from_row":m.A/cols,"from_col":m.A%cols,"to_row":m.C/cols,"to_col":m.C%cols}}

// iterate runs iterative deepening over the root moves until the deadline, the stop flag or max_depth.
func (s *Searcher) iterate(p Position,ordered []Move,startDepth int)(best Move,bestValue float64,done int,timedOut bool){
	best,bestValue=ordered[0],math.NaN()
	for depth:=startDepth;depth<=s.MaxDepth;depth++{
		if time.Now().After(s.Deadline)||s.Stop.Load(){timedOut=true;break}
		vals:=make([]float64,len(ordered));alpha:=math.Inf(-1);complete:=true
		for i,m:=range ordered{v,ok:=safeRootWindow(s,p,m,depth,alpha,math.Inf(1));if !ok{complete=false;break};vals[i]=v;if v>alpha{alpha=v}}
		if !complete{timedOut=true;break};idx:=0;for i:=1;i<len(vals);i++{if vals[i]>vals[idx]{idx=i}};best,bestValue,done=ordered[idx],vals[idx],depth;ordered=append([]Move{best},withoutMove(ordered,best)...);if s.OnDepth!=nil{s.OnDepth(best,bestValue,done)}
	}
	return
}
// iterateParallel splits each depth's root moves across workers: the main worker searches the previous best
// move first to set alpha, then all workers take the remaining moves from a shared queue with the best alpha
// so far. Workers keep their searchers (eval cache, buffers) across depths and share one transposition table.
func iterateParallel(p Position,ordered []Move,workers []*Searcher)(best Move,bestValue float64,done int,timedOut bool){
	main:=workers[0];best,bestValue=ordered[0],math.NaN()
	for depth:=1;depth<=main.MaxDepth;depth++{
		if time.Now().After(main.Deadline){timedOut=true;break}
		vals:=make([]float64,len(ordered));first,ok:=safeRootWindow(main,p,ordered[0],depth,math.Inf(-1),math.Inf(1));if !ok{timedOut=true;break};vals[0]=first
		var alphaBits atomic.Uint64;alphaBits.Store(math.Float64bits(first));var next atomic.Int64;next.Store(1);var failed atomic.Bool;var wg sync.WaitGroup
		for _,w:=range workers{wg.Add(1);go func(w *Searcher){defer wg.Done();for{i:=int(next.Add(1)-1);if i>=len(ordered)||failed.Load(){return};alpha:=math.Float64frombits(alphaBits.Load());v,ok:=safeRootWindow(w,p,ordered[i],depth,alpha,math.Inf(1));if !ok{failed.Store(true);return};vals[i]=v
			for{old:=alphaBits.Load();if v<=math.Float64frombits(old)||alphaBits.CompareAndSwap(old,math.Float64bits(v)){break}}}}(w)}
		wg.Wait();if failed.Load(){timedOut=true;break}
		idx:=0;for i:=1;i<len(vals);i++{if vals[i]>vals[idx]{idx=i}};best,bestValue,done=ordered[idx],vals[idx],depth;ordered=append([]Move{best},withoutMove(ordered,best)...);if main.OnDepth!=nil{main.OnDepth(best,bestValue,done)}
	}
	return
}
// winProb converts a search value back to the estimated win probability (as the Python searcher reports).
func (s *Searcher) winProb(p *Position,value float64) float64 {v:=value/math.Pow(s.estMultiplier(p),s.Lam);if v<0{v/=s.Kappa};return math.Max(0,math.Min(1,(v+1)/2))}
func choose(p Position,req Request,emit func(any))(map[string]any,map[string]any){
	if p.NoCapturePlies >= p.NoCaptureLimit { return nil, map[string]any{"source":"rules","draw":true,"no_capture_plies":p.NoCapturePlies} }
	strategy:=16;if strings.Contains(req.Strategy,"17"){strategy=17};threads:=req.Threads;if threads<1{threads=1};if threads>runtime.NumCPU(){threads=runtime.NumCPU()};budget:=req.TimeBudgetMS;if budget<1{budget=1};deadline:=time.Now().Add(time.Duration(budget)*time.Millisecond)
	mainSearcher:=makeSearcher(req.Params,strategy,deadline,nil,nil)
	// The deadline also raises the stop flag from a timer, in addition to the clock checks in tick().
	stopTimer:=time.AfterFunc(time.Until(deadline),func(){mainSearcher.Stop.Store(true)});defer stopTimer.Stop()
	if req.Progress&&emit!=nil{mainSearcher.OnDepth=func(m Move,v float64,d int){emit(map[string]any{"progress":map[string]any{"action":rootAction(m),"info":map[string]any{"source":"search","depth":d,"score":v,"win_prob":mainSearcher.winProb(&p,v),"multiplier":p.multiplier(),"partial":true}}})}}
	ordered:=append([]Move(nil),mainSearcher.orderedMoves(&p,0,Move{},false)...);if len(ordered)==0{return nil,map[string]any{"source":"search","depth":0,"nodes":0}}
	var best Move;var bestValue float64;var done int;var timedOut bool;nodes:=int64(0)
	if threads==1||len(ordered)==1{best,bestValue,done,timedOut=mainSearcher.iterate(p,ordered,1);nodes=mainSearcher.Nodes}else{
		workers:=[]*Searcher{mainSearcher};for i:=1;i<min(threads,len(ordered));i++{workers=append(workers,makeSearcher(req.Params,strategy,deadline,mainSearcher.TT,mainSearcher.Stop))}
		best,bestValue,done,timedOut=iterateParallel(p,ordered,workers);for _,w:=range workers{nodes+=w.Nodes}
	}
	info:=map[string]any{"source":"search","depth":done,"nodes":nodes,"timed_out":timedOut,"threads":threads,"multiplier":p.multiplier()};if !math.IsNaN(bestValue){info["score"]=bestValue;info["win_prob"]=mainSearcher.winProb(&p,bestValue)};return rootAction(best),info
}
func withoutMove(moves []Move,best Move)[]Move{out:=make([]Move,0,len(moves)-1);skipped:=false;for _,m:=range moves{if !skipped&&m==best{skipped=true;continue};out=append(out,m)};return out}

func handle(req Request,emit func(any)) Response {
	if req.Strategy!="StrategicV16AI"&&req.Strategy!="StrategicV17AI"{return Response{Error:"unsupported strategy; choose StrategicV16AI or StrategicV17AI"}}
	p,err:=parsePosition(req.State);if err!=nil{return Response{Error:err.Error()}}
	action,info:=choose(p,req,emit);return Response{Action:action,Info:info}
}
func main(){scanner:=bufio.NewScanner(os.Stdin);scanner.Buffer(make([]byte,4096),8*1024*1024);writer:=bufio.NewWriter(os.Stdout);defer writer.Flush();for scanner.Scan(){var req Request;if err:=json.Unmarshal(scanner.Bytes(),&req);err!=nil{_ = json.NewEncoder(writer).Encode(Response{Error:err.Error()});writer.Flush();continue};emit:=func(v any){_ = json.NewEncoder(writer).Encode(v);writer.Flush()}
		// Safety net: if a request overruns its budget by 3s, report every goroutine stack and exit instead of hanging.
		watchdog:=time.AfterFunc(time.Duration(max(1,req.TimeBudgetMS)+3000)*time.Millisecond,func(){buf:=make([]byte,1<<20);n:=runtime.Stack(buf,true);fmt.Fprintf(os.Stderr,"engine watchdog: search exceeded its time budget by 3s\n%s",buf[:n]);os.Exit(3)})
		resp:=handle(req,emit);watchdog.Stop();_ = json.NewEncoder(writer).Encode(resp);writer.Flush()}}
