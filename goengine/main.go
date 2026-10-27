package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
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
)

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
	Flip float64
	Winner int
	NoCapturePlies int
	ChaseCount [2]int
	ChaseTarget [2]int
	LongChaseLimit int
	NoCaptureLimit int
}
type Undo struct {
	A, C int
	PA, PC int8
	HP [2]float64
	Turn int
	Cur, Max [2]int
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
	return p, nil
}

func (p *Position) gen() (caps, hiddenCaps, moves, flips []Move) {
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
	legalMoves := make([]Move, 0, len(moves))
	for _, m := range moves {
		u := p.doMove(m.A, m.C)
		violates := p.ChaseCount[p.Turn^1] >= p.LongChaseLimit-1
		p.undo(u)
		if !violates { legalMoves = append(legalMoves, m) }
	}
	moves = legalMoves
	return
}

func (p *Position) captures() []Move {
	caps, _, _, _ := p.gen()
	return caps
}
func (p *Position) save(a,c,poolCode int) Undo {
	return Undo{A:a,C:c,PA:p.B[a],PC:p.B[c],HP:p.HP,Turn:p.Turn,Cur:p.Cur,Max:p.Max,Flip:p.Flip,Winner:p.Winner,NoCapturePlies:p.NoCapturePlies,ChaseCount:p.ChaseCount,ChaseTarget:p.ChaseTarget,PoolCode:poolCode}
}
func (p *Position) undo(u Undo) {
	p.B[u.A], p.B[u.C] = u.PA, u.PC
	p.HP, p.Turn, p.Cur, p.Max, p.Flip, p.Winner = u.HP, u.Turn, u.Cur, u.Max, u.Flip, u.Winner
	p.NoCapturePlies, p.ChaseCount, p.ChaseTarget = u.NoCapturePlies, u.ChaseCount, u.ChaseTarget
	if u.PoolCode > 0 { p.Pool[u.PoolCode]++ }
}
func (p *Position) hit(side int) { p.Cur[side]++; if p.Cur[side] > p.Max[side] { p.Max[side] = p.Cur[side] } }
func (p *Position) checkOver() {
	if p.HP[red] <= 0 { p.Winner = black } else if p.HP[black] <= 0 { p.Winner = red }
}
func (p *Position) doMove(a,c int) Undo {
	u := p.save(a,c,0); attacker, target := int(p.B[a]), int(p.B[c]); p.B[c],p.B[a] = p.B[a],empty
	if target != empty { p.HP[sideOf[target]] -= float64(hpc[target]); p.hit(p.Turn); p.checkOver()
	} else { p.Cur[p.Turn] = 0 }
	me, opponent := p.Turn, p.Turn^1
	if target == empty { p.NoCapturePlies++ } else { p.NoCapturePlies = 0 }
	if target != empty {
		for side := 0; side < 2; side++ { if p.ChaseTarget[side] == c { p.ChaseTarget[side], p.ChaseCount[side] = -1, 0 } }
	}
	if p.ChaseTarget[opponent] == a { p.ChaseTarget[opponent] = c }
	if target != empty { p.ChaseTarget[me], p.ChaseCount[me] = -1, 0
	} else {
		targets := p.captureTargetsFrom(c, me); old := p.ChaseTarget[me]; found := false
		for _, sq := range targets { if sq == old { p.ChaseCount[me]++; found = true; break } }
		if !found { if len(targets) > 0 { p.ChaseTarget[me], p.ChaseCount[me] = targets[0], 1 } else { p.ChaseTarget[me], p.ChaseCount[me] = -1, 0 } }
	}
	p.Turn ^= 1
	_ = attacker
	return u
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
	u := p.save(sq,sq,code); p.B[sq] = int8(code); p.Pool[code]--; p.ChaseCount[p.Turn],p.ChaseTarget[p.Turn]=0,-1; p.Cur[p.Turn] = 0; p.Turn ^= 1; return u
}
func (p *Position) doFlipApprox(sq int) Undo {
	u := p.save(sq,sq,0); p.ChaseCount[p.Turn],p.ChaseTarget[p.Turn]=0,-1; p.Cur[p.Turn] = 0; p.Turn ^= 1; return u
}
func (p *Position) doHiddenCapture(a,c,code int) Undo {
	u := p.save(a,c,code); p.Pool[code]--; p.HP[sideOf[code]] -= float64(hpc[code]); p.B[c] = p.B[a]; p.B[a] = empty
	me,opponent:=p.Turn,p.Turn^1; p.hit(me); p.checkOver(); p.NoCapturePlies=0; if p.ChaseTarget[opponent]==a {p.ChaseTarget[opponent]=c}; p.ChaseTarget[me],p.ChaseCount[me]=-1,0; p.Turn ^= 1; return u
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
func (p *Position) comboFactor() int { total:=0; for _,m:=range p.Max {if m>3 {total+=m}}; if total==0{return 1}; return total }
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
type Searcher struct {
	P map[string]any
	W Weights
	PV [7]float64
	Deadline time.Time
	Nodes int64
	TT map[uint64]TTEntry
	Strategy int
	Interval int64
}
type TTEntry struct { Depth int; Value float64; Flag uint8; Best Move; HasBest bool }
type scoredMove struct { Score float64; M Move }
type timeoutPanic struct{}

func number(v any, fallback float64) float64 { if n,ok:=v.(float64); ok{return n}; if n,ok:=v.(int);ok{return float64(n)};return fallback }
func intParam(p map[string]any,key string,fallback int) int { return int(number(p[key],float64(fallback))) }
func floatParam(p map[string]any,key string,fallback float64) float64 { return number(p[key],fallback) }
func weightMap(p map[string]any) map[string]any { if w,ok:=p["weights"].(map[string]any);ok{return w};return nil }
func defaultParams() map[string]any { return map[string]any{"multiplier_exponent":1.0,"loss_aversion":1.0,"win_temperature":12.0,"max_depth":8,"root_flips":8,"node_flips":3,"node_quiet_moves":10,"qsearch_depth":4,"full_chance_plies":1,"q_hidden_caps":1,"q_hidden_depth":1,"time_check_interval":32,"early_pool_threshold":20,"mid_pool_threshold":12,"early_root_flips":6,"mid_root_flips":7,"early_node_flips":2,"mid_node_flips":3,"early_chance_samples":6,"mid_chance_samples":8} }
func makeSearcher(params map[string]any,strategy int,deadline time.Time) *Searcher {
	d:=defaultParams(); for k,v:=range params {d[k]=v}; p:=d
	wm:=weightMap(p); nested:=map[string]any{}; if raw,ok:=wm["piece_value"].(map[string]any);ok{nested=raw}
	w:=Weights{HP:1,Mobility:0.3,Hanging:0.6,HangingOne:0.15,Pool:0.3,KingSoldierAdj:8,KingSoldierNear:3,KingCannon:5,KingTrapped:2,Tempo:0.5}
	if wm!=nil {w.HP=number(wm["hp"],w.HP);w.Mobility=number(wm["mobility"],w.Mobility);w.Hanging=number(wm["hanging"],w.Hanging);w.HangingOne=number(wm["hanging_one"],w.HangingOne);w.Pool=number(wm["pool"],w.Pool);w.KingSoldierAdj=number(wm["king_soldier_adj"],w.KingSoldierAdj);w.KingSoldierNear=number(wm["king_soldier_near"],w.KingSoldierNear);w.KingCannon=number(wm["king_cannon"],w.KingCannon);w.KingTrapped=number(wm["king_trapped"],w.KingTrapped);w.Tempo=number(wm["tempo"],w.Tempo)}
	pv:= [7]float64{10,6,3,2.5,2,3,1.5}; for i,t:=range pieceTypes {pv[i]=number(nested[t],pv[i])}
	interval:=intParam(p,"time_check_interval",32); if strategy==16 {interval=256}; if interval<1{interval=1}
	return &Searcher{P:p,W:w,PV:pv,Deadline:deadline,TT:make(map[uint64]TTEntry,1<<15),Strategy:strategy,Interval:int64(interval)}
}

func (s *Searcher) tick() { s.Nodes++; if s.Nodes%s.Interval==0 && time.Now().After(s.Deadline) {panic(timeoutPanic{})} }
func (s *Searcher) staticEval(p *Position) float64 {
	me,op:=p.Turn,p.Turn^1; var material [2]float64; var mobility [2]float64; kings:=[2]int{-1,-1}; soldiers:=[2][]int{}; cannons:=[2][]int{}; threatened:=[]float64{}
	for sq,raw:=range p.B { code:=int(raw);if code==empty||code==hidden{continue}; side,t:=sideOf[code],typeOf[code];material[side]+=s.PV[t];if t==king{kings[side]=sq}else if t==soldier{soldiers[side]=append(soldiers[side],sq)}else if t==cannon{cannons[side]=append(cannons[side],sq)};if t!=cannon {for _,n:=range neighbors[sq]{if p.B[n]==empty{mobility[side]++}}};if side==me&&p.attacked(sq,code){threatened=append(threatened,float64(hpc[code])+s.PV[t])}}
	score:=s.W.HP*(p.HP[me]-p.HP[op])+material[me]-material[op]+s.W.Mobility*(mobility[me]-mobility[op])
	if len(threatened)>0 {sum,biggest:=0.0,0.0;for _,v:=range threatened{sum+=v;if v>biggest{biggest=v}};score-=s.W.Hanging*(sum-biggest)+s.W.HangingOne*biggest}
	var poolValue [2]float64;for c:=1;c<=14;c++{if p.Pool[c]>0{poolValue[sideOf[c]]+=float64(p.Pool[c])*s.PV[typeOf[c]]}};score+=s.W.Pool*(poolValue[me]-poolValue[op])
	for side:=0;side<2;side++{k:=kings[side];if k<0{continue};enemy:=side^1;danger:=0.0;for _,sq:=range soldiers[enemy]{d:=distance[k][sq];if d==1{danger+=s.W.KingSoldierAdj}else if d==2{danger+=s.W.KingSoldierNear}};for _,sq:=range cannons[enemy]{if p.cannonHits(sq,k){danger+=s.W.KingCannon}};free:=0;for _,n:=range neighbors[k]{if p.B[n]==empty{free++}};if free<2{danger+=s.W.KingTrapped*float64(2-free)};if side==me{score-=danger}else{score+=danger}}
	return score+s.W.Tempo
}
func (s *Searcher) estMultiplier(p *Position) float64 {m:=float64(p.comboFactor())*(p.Flip+p.hpDiff());if m<1{return 1};return m}
func (s *Searcher) objective(v,m float64) float64 {v*=math.Pow(m,floatParam(s.P,"multiplier_exponent",1));if v<0{v*=floatParam(s.P,"loss_aversion",1)};return v}
func (s *Searcher) leafValue(p *Position) float64 {x:=s.staticEval(p)/floatParam(s.P,"win_temperature",12);if x>30{x=30}else if x< -30{x=-30};prob:=2/(1+math.Exp(-x))-1;return s.objective(prob,s.estMultiplier(p))}
func (s *Searcher) terminalValue(p *Position,ply int) float64 {m:=math.Max(1,p.multiplier());v:=(1+0.01*math.Log(m))*(1-0.001*float64(ply));if p.Winner!=p.Turn{v=-v};return s.objective(v,m)}

func (s *Searcher) quietScore(p *Position,a,c int) float64 {
	piece:=int(p.B[a]);score:=0.0;if p.attacked(a,piece){score+=float64(hpc[piece])*10+s.PV[typeOf[piece]]*10};u:=p.doMove(a,c);defer p.undo(u);if p.attacked(c,piece){score-=(float64(hpc[piece])+s.PV[typeOf[piece]])*10};for _,n:=range neighbors[c]{q:=int(p.B[n]);if q!=empty&&q!=hidden&&canEat[piece][q]{score+=float64(hpc[q])*3}};return score
}
func (s *Searcher) flipFull(p *Position,sq int,outcomes []int,total int) float64 {
	me:=p.Turn; own,enemy:=[]int{},[]int{};for _,n:=range neighbors[sq]{q:=int(p.B[n]);if q!=empty&&q!=hidden{if sideOf[q]==me{own=append(own,q)}else{enemy=append(enemy,q)}}};score:=0.0
	for _,code:=range outcomes{prob:=float64(p.Pool[code])/float64(total);t:=typeOf[code];gain:=0.0;if sideOf[code]==me{gain=s.PV[t];for _,e:=range enemy{if typeOf[e]!=cannon&&canEat[e][code]{gain-=float64(hpc[code])+s.PV[t];break}}}else{gain=-s.PV[t];if t!=cannon{for _,o:=range own{if canEat[code][o]{gain-=float64(hpc[o])+s.PV[typeOf[o]];break}}}};score+=prob*gain};return score
}
func (s *Searcher) flipFast(p *Position,sq int) float64 {score:=0.0;for _,n:=range neighbors[sq]{q:=int(p.B[n]);if q!=empty&&q!=hidden{if sideOf[q]==p.Turn{score++}else{score-=1+float64(ranks[typeOf[q]])}}};return score}
func (s *Searcher) phaseLimits(p *Position)(int,int){pool:=0;for _,n:=range p.Pool{pool+=n};if s.Strategy!=17{return intParam(s.P,"root_flips",8),intParam(s.P,"node_flips",3)};if pool>=intParam(s.P,"early_pool_threshold",20){return intParam(s.P,"early_root_flips",6),intParam(s.P,"early_node_flips",2)};if pool>=intParam(s.P,"mid_pool_threshold",12){return intParam(s.P,"mid_root_flips",7),intParam(s.P,"mid_node_flips",3)};return intParam(s.P,"root_flips",8),intParam(s.P,"node_flips",3)}
func (s *Searcher) orderedMoves(p *Position,ply int,ttBest Move,hasTT bool) []Move {
	caps,hcaps,moves,flips:=p.gen();scored:=[]scoredMove{};root:=ply==0
	for _,m:=range caps{v:=int(p.B[m.C]);att:=int(p.B[m.A]);scored=append(scored,scoredMove{10000+float64(hpc[v])*20+s.PV[typeOf[v]]*5-float64(ranks[typeOf[att]]),m})}
	if len(hcaps)>0{total:=0;for _,n:=range p.Pool{total+=n};enemy:=p.Turn^1;expected:=0.0;for c:=1;c<=14;c++{if p.Pool[c]>0{sign:=-1.0;if sideOf[c]==enemy{sign=1};expected+=float64(p.Pool[c]*hpc[c])*sign}};if total>0{expected/=float64(total)};for _,m:=range hcaps{scored=append(scored,scoredMove{5000+expected*20,m})}}
	quiet:=make([]scoredMove,0,len(moves));for _,m:=range moves{quiet=append(quiet,scoredMove{s.quietScore(p,m.A,m.C),m})};sort.SliceStable(quiet,func(i,j int)bool{return quiet[i].Score>quiet[j].Score});if !root{limit:=intParam(s.P,"node_quiet_moves",10);if limit<0{limit=0};if len(quiet)>limit{quiet=quiet[:limit]}};scored=append(scored,quiet...)
	rootLimit,nodeLimit:=s.phaseLimits(p);flipScores:=make([]scoredMove,0,len(flips));if len(flips)>0{outcomes,total:=[]int{},0;for c:=1;c<=14;c++{if p.Pool[c]>0{outcomes=append(outcomes,c);total+=p.Pool[c]}};for _,m:=range flips{v:=s.flipFast(p,m.A);if root&&total>0{v=s.flipFull(p,m.A,outcomes,total)};if root{v-=1000};flipScores=append(flipScores,scoredMove{v,m})};sort.SliceStable(flipScores,func(i,j int)bool{return flipScores[i].Score>flipScores[j].Score});limit:=nodeLimit;if root{limit=rootLimit};if limit<len(flipScores){flipScores=flipScores[:max(0,limit)]}}
	sort.SliceStable(scored,func(i,j int)bool{return scored[i].Score>scored[j].Score});base:=make([]Move,0,len(scored)+len(flipScores));for _,x:=range scored{base=append(base,x.M)}
	if s.Strategy==17{tactical,quietMoves:=[]Move{},[]Move{};for _,m:=range base{if m.Kind==captureKind||m.Kind==hiddenCaptureKind{tactical=append(tactical,m)}else{quietMoves=append(quietMoves,m)}};ordered:=append([]Move{},tactical...);for i:=0;i<max(len(flipScores),len(quietMoves));i++{if i<len(flipScores){ordered=append(ordered,flipScores[i].M)};if i<len(quietMoves){ordered=append(ordered,quietMoves[i])}};if hasTT{for i,m:=range ordered{if m==ttBest{ordered=append([]Move{m},append(ordered[:i],ordered[i+1:]...)...);break}}};return ordered}
	base=append(base,make([]Move,len(flipScores))...);for i,x:=range flipScores{base[len(scored)+i]=x.M};if hasTT{for i,m:=range base{if m==ttBest{base=append([]Move{m},append(base[:i],base[i+1:]...)...);break}}};return base
}

func (p *Position) key() uint64 {
	h:=uint64(1469598103934665603);mix:=func(v uint64){h^=v;h*=1099511628211};for _,v:=range p.B{mix(uint64(v))};for _,v:=range p.Pool{mix(uint64(v))};for _,v:=range p.HP{mix(math.Float64bits(v))};mix(uint64(p.Turn));for _,v:=range p.Cur{mix(uint64(v))};for _,v:=range p.Max{mix(uint64(v))};mix(math.Float64bits(p.Flip));mix(uint64(p.Winner+1));mix(uint64(p.NoCapturePlies));for _,v:=range p.ChaseCount{mix(uint64(v))};for _,v:=range p.ChaseTarget{mix(uint64(v+1))};mix(uint64(p.LongChaseLimit));mix(uint64(p.NoCaptureLimit));return h
}

func (s *Searcher) chanceWeights(p *Position) map[int]float64 {
	total:=0;for _,n:=range p.Pool{total+=n};if total==0{return map[int]float64{}}
	n:=0;if s.Strategy==17{if total>=intParam(s.P,"early_pool_threshold",20){n=intParam(s.P,"early_chance_samples",6)}else if total>=intParam(s.P,"mid_pool_threshold",12){n=intParam(s.P,"mid_chance_samples",8)}}
	if n<=0 {w:=map[int]float64{};for c:=1;c<=14;c++{if p.Pool[c]>0{w[c]=float64(p.Pool[c])/float64(total)}};return w}
	protected,remaining:=[]int{},[]int{};remainingSize:=0
	for c:=1;c<=14;c++{if p.Pool[c]==0{continue};if typeOf[c]==king||typeOf[c]==soldier{protected=append(protected,c)}else{remaining=append(remaining,c);remainingSize+=p.Pool[c]}}
	w:=map[int]float64{};for _,c:=range protected{w[c]=float64(p.Pool[c])/float64(total)};if remainingSize==0{return w};if remainingSize<=n{for _,c:=range remaining{w[c]=float64(p.Pool[c])/float64(total)};return w}
	h:=fnv.New64a();for _,v:=range p.B{h.Write([]byte{byte(v)})};for _,v:=range p.Pool{h.Write([]byte{byte(v)})};buf:=make([]byte,8);binary.LittleEndian.PutUint64(buf,math.Float64bits(p.Flip));h.Write(buf);offset:=rand.New(rand.NewSource(int64(h.Sum64()))).Float64();sampled:=map[int]int{};point:=0
	for i:=0;i<n;i++{x:=(float64(i)+offset)*float64(remainingSize)/float64(n);cum:=0;for _,c:=range remaining{cum+=p.Pool[c];if x<float64(cum){sampled[c]++;break}};point++};_ = point
	mass:=float64(remainingSize)/float64(total);for c,k:=range sampled{w[c]+=mass*float64(k)/float64(n)};return w
}
func sortedCodes(weights map[int]float64) []int { codes:=make([]int,0,len(weights));for c:=range weights{codes=append(codes,c)};sort.Ints(codes);return codes }
func (s *Searcher) child(p *Position,m Move,depth int,alpha,beta float64,ply int) float64 {
	if m.Kind==moveKind||m.Kind==captureKind{u:=p.doMove(m.A,m.C);defer p.undo(u);return -s.negamax(p,depth-1,-beta,-alpha,ply+1)}
	if m.Kind==flipKind&&ply>=intParam(s.P,"full_chance_plies",1){u:=p.doFlipApprox(m.A);defer p.undo(u);return -s.negamax(p,depth-1,-beta,-alpha,ply+1)}
	weights:=s.chanceWeights(p);value:=0.0;for _,code:=range sortedCodes(weights){prob:=weights[code];var u Undo;if m.Kind==flipKind{u=p.doFlip(m.A,code)}else{u=p.doHiddenCapture(m.A,m.C,code)};v:=-s.negamax(p,depth-1,math.Inf(-1),math.Inf(1),ply+1);p.undo(u);value+=prob*v};return value
}
func poolSize(p *Position)int{n:=0;for _,v:=range p.Pool{n+=v};return n}

func (s *Searcher) negamax(p *Position,depth int,alpha,beta float64,ply int) float64 {
	s.tick();if p.NoCapturePlies>=p.NoCaptureLimit{return 0};if p.Winner>=0{return s.terminalValue(p,ply)};if depth<=0{return s.qsearch(p,alpha,beta,ply,intParam(s.P,"qsearch_depth",4),intParam(s.P,"q_hidden_depth",1))}
	key:=p.key();entry,ok:=s.TT[key];ttBest,hasTT:=entry.Best,ok&&entry.HasBest;alpha0:=alpha;if ok&&entry.Depth>=depth{if entry.Flag==0{return entry.Value};if entry.Flag==1&&entry.Value>=beta{return entry.Value};if entry.Flag==2&&entry.Value<=alpha{return entry.Value}}
	moves:=s.orderedMoves(p,ply,ttBest,hasTT);if len(moves)==0{return s.leafValue(p)};best:=-math.MaxFloat64;var bestMove Move;hasBest:=false
	for _,m:=range moves{v:=s.child(p,m,depth,alpha,beta,ply);if v>best{best,bestMove,hasBest=v,m,true};if v>alpha{alpha=v};if alpha>=beta{break}}
	flag:=uint8(0);if best<=alpha0{flag=2}else if best>=beta{flag=1};s.TT[key]=TTEntry{Depth:depth,Value:best,Flag:flag,Best:bestMove,HasBest:hasBest};return best
}
func (s *Searcher) qsearch(p *Position,alpha,beta float64,ply,qdepth,hiddenLeft int)float64{
	s.tick();if p.NoCapturePlies>=p.NoCaptureLimit{return 0};if p.Winner>=0{return s.terminalValue(p,ply)};stand:=s.leafValue(p);if qdepth<=0||stand>=beta{return stand};if stand>alpha{alpha=stand}
	caps:=p.captures();sort.SliceStable(caps,func(i,j int)bool{a,b:=caps[i],caps[j];return hpc[int(p.B[b.C])]*20-ranks[typeOf[int(p.B[b.A])]]>hpc[int(p.B[a.C])]*20-ranks[typeOf[int(p.B[a.A])]]})
	for _,m:=range caps{u:=p.doMove(m.A,m.C);v:=-s.qsearch(p,-beta,-alpha,ply+1,qdepth-1,hiddenLeft);p.undo(u);if v>=beta{return v};if v>alpha{alpha=v}}
	if s.Strategy==17&&hiddenLeft>0{_,hc,_,_:=p.gen();if len(hc)>0&&poolSize(p)>0{likely:=0;for c:=1;c<=14;c++{if p.Pool[c]>p.Pool[likely]{likely=c}};type candidate struct{m Move;score float64};cs:=[]candidate{};for _,m:=range hc{u:=p.doHiddenCapture(m.A,m.C,likely);score:=-s.leafValue(p);if p.Winner>=0{score=-s.terminalValue(p,ply+1)};p.undo(u);cs=append(cs,candidate{m,score})};sort.SliceStable(cs,func(i,j int)bool{return cs[i].score>cs[j].score});limit:=intParam(s.P,"q_hidden_caps",1);if limit<0{limit=0};if limit>len(cs){limit=len(cs)};for _,candidate:=range cs[:limit]{value:=0.0;weights:=s.chanceWeights(p);for _,code:=range sortedCodes(weights){prob:=weights[code];u:=p.doHiddenCapture(candidate.m.A,candidate.m.C,code);v:=-s.qsearch(p,math.Inf(-1),math.Inf(1),ply+1,qdepth-1,hiddenLeft-1);p.undo(u);value+=prob*v};if value>=beta{return value};if value>alpha{alpha=value}}}}
	return alpha
}

func safeRootWindow(search *Searcher,p Position,m Move,depth int,alpha,beta float64)(value float64,ok bool){
	defer func(){if r:=recover();r!=nil{if _,yes:=r.(timeoutPanic);yes{ok=false}else{panic(r)}}}()
	value=search.child(&p,m,depth,alpha,beta,0);return value,true
}
func safeRoot(search *Searcher,p Position,m Move,depth int)(float64,bool){return safeRootWindow(search,p,m,depth,math.Inf(-1),math.Inf(1))}
func rootAction(m Move)map[string]any{if m.Kind==flipKind{return map[string]any{"type":"flip","row":m.A/cols,"col":m.A%cols}};kind:="move";if m.Kind==captureKind||m.Kind==hiddenCaptureKind{kind="capture"};return map[string]any{"type":kind,"from_row":m.A/cols,"from_col":m.A%cols,"to_row":m.C/cols,"to_col":m.C%cols}}

func parallelDepth(base Position,moves []Move,depth,threads,strategy int,params map[string]any,deadline time.Time)([]float64,int64,bool){
	if threads<1{threads=1};if threads>len(moves){threads=len(moves)};jobs:=make(chan int,len(moves));results:=make(chan struct{i int;v float64;nodes int64;ok bool},len(moves));for i:=range moves{jobs<-i};close(jobs);var wg sync.WaitGroup
	for w:=0;w<threads;w++{wg.Add(1);go func(){defer wg.Done();search:=makeSearcher(params,strategy,deadline);for i:=range jobs{before:=search.Nodes;v,ok:=safeRoot(search,base,moves[i],depth);results<-struct{i int;v float64;nodes int64;ok bool}{i,v,search.Nodes-before,ok}}}()}
	go func(){wg.Wait();close(results)}();values:=make([]float64,len(moves));var nodes int64;complete:=true;received:=0;for r:=range results{received++;nodes+=r.nodes;if r.ok{values[r.i]=r.v}else{complete=false}};if received!=len(moves){complete=false};return values,nodes,complete
}
func choose(p Position,req Request)(map[string]any,map[string]any){
	if p.NoCapturePlies >= p.NoCaptureLimit { return nil, map[string]any{"source":"rules","draw":true,"no_capture_plies":p.NoCapturePlies} }
	strategy:=16;if strings.Contains(req.Strategy,"17"){strategy=17};threads:=req.Threads;if threads<1{threads=1};if threads>runtime.NumCPU(){threads=runtime.NumCPU()};budget:=req.TimeBudgetMS;if budget<1{budget=1};deadline:=time.Now().Add(time.Duration(budget)*time.Millisecond);rootSearcher:=makeSearcher(req.Params,strategy,deadline)
	ordered:=rootSearcher.orderedMoves(&p,0,Move{},false);if len(ordered)==0{return nil,map[string]any{"source":"search","depth":0,"nodes":0}}
	best:=ordered[0];bestValue:=math.NaN();done:=0;nodes:=int64(0);timedOut:=false;maxDepth:=intParam(rootSearcher.P,"max_depth",8)
	for depth:=1;depth<=maxDepth;depth++{if time.Now().After(deadline){timedOut=true;break};var vals []float64;var work int64;complete:=true
		if threads==1{vals=make([]float64,len(ordered));alpha:=math.Inf(-1);for i,m:=range ordered{v,ok:=safeRootWindow(rootSearcher,p,m,depth,alpha,math.Inf(1));nodes+=rootSearcher.Nodes-work;work=rootSearcher.Nodes;if !ok{complete=false;break};vals[i]=v;if v>alpha{alpha=v}}}else{vals,work,complete=parallelDepth(p,ordered,depth,threads,strategy,rootSearcher.P,deadline);nodes+=work}
		if !complete{timedOut=true;break};idx:=0;for i:=1;i<len(vals);i++{if vals[i]>vals[idx]{idx=i}};best,bestValue,done=ordered[idx],vals[idx],depth;ordered=append([]Move{best},withoutMove(ordered,best)...)
	}
	info:=map[string]any{"source":"search","depth":done,"nodes":nodes,"timed_out":timedOut,"threads":threads,"multiplier":p.multiplier()};if !math.IsNaN(bestValue){info["score"]=bestValue};return rootAction(best),info
}
func withoutMove(moves []Move,best Move)[]Move{out:=make([]Move,0,len(moves)-1);skipped:=false;for _,m:=range moves{if !skipped&&m==best{skipped=true;continue};out=append(out,m)};return out}

func handle(req Request) Response {
	if req.Strategy!="StrategicV16AI"&&req.Strategy!="StrategicV17AI"{return Response{Error:"unsupported strategy; choose StrategicV16AI or StrategicV17AI"}}
	p,err:=parsePosition(req.State);if err!=nil{return Response{Error:err.Error()}}
	action,info:=choose(p,req);return Response{Action:action,Info:info}
}
func main(){scanner:=bufio.NewScanner(os.Stdin);scanner.Buffer(make([]byte,4096),8*1024*1024);writer:=bufio.NewWriter(os.Stdout);defer writer.Flush();for scanner.Scan(){var req Request;if err:=json.Unmarshal(scanner.Bytes(),&req);err!=nil{_ = json.NewEncoder(writer).Encode(Response{Error:err.Error()});writer.Flush();continue};resp:=handle(req);_ = json.NewEncoder(writer).Encode(resp);writer.Flush()}}
