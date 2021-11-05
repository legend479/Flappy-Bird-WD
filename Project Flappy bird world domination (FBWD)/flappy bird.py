from operator import xor
from typing import DefaultDict
from neat import config, population
import pygame
import neat
import random
import time
import os
import pickle
pygame.font.init()

WIN_WIDTH=500
WIN_HEIGHT=650

GEN=0

Scale_factor=WIN_HEIGHT/400

BIRD_IMGS=[pygame.transform.rotozoom(pygame.image.load(os.path.join("imgs","bird1.png")),0,Scale_factor),
            pygame.transform.rotozoom(pygame.image.load(os.path.join("imgs","bird2.png")),0,Scale_factor),
            pygame.transform.rotozoom(pygame.image.load(os.path.join("imgs","bird3.png")),0,Scale_factor)]

PIPE_IMG=pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")))

BASE_IMG=pygame.transform.scale(pygame.image.load(os.path.join("imgs","base.png")),(WIN_WIDTH,WIN_HEIGHT/6))

BG_IMG=pygame.transform.scale(pygame.image.load(os.path.join("imgs","bg.png")),(WIN_WIDTH,WIN_HEIGHT))

STAT_FONT=pygame.font.SysFont("comic sans",36,True)

class Bird:
    IMGS=BIRD_IMGS
    MAX_ROTATION=25
    ROT_VEL=20
    ANIMATION_TIME=3
    JUMP_FORCE=-7
    GRAVITY=1
    TERMINAL_VEL=16

    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.tilt=0
        self.tick_count=0
        self.vel=0
        self.height=self.y
        self.img_count=0
        self.img=self.IMGS[0]

    
    def Jump(self):
        self.vel=self.JUMP_FORCE
        self.tick_count=0
        self.height=self.y

    def Move(self):
        self.tick_count+=1

        d= (self.vel*self.tick_count) +(self.GRAVITY*(self.tick_count**2))

        if d>=self.TERMINAL_VEL:
            d=self.TERMINAL_VEL
        #Giving the bird a boosted jump(bumppy jump instead of smooth)
        elif d<0:
            d-=2
        
        self.y+=d

        if d<0 :#Check if removing second condition makes a difference
             self.tilt=self.MAX_ROTATION# Check if removing the 2nd conditional affects the code
        elif d>0:
           if self.tilt>-90:
              self.tilt-=self.ROT_VEL
        
    def Draw(self,surface):
        self.img_count+=1

        n=(self.img_count//self.ANIMATION_TIME)

        self.img=self.IMGS[round(n%3)]

        if self.tilt<=-80:
            self.img=self.IMGS[1]
            self.img_count=self.ANIMATION_TIME*2
        
        rotated_img=pygame.transform.rotate(self.img,self.tilt)
        new_rect=rotated_img.get_rect(center=self.img.get_rect(topleft = (self.x , self.y)).center)

        surface.blit(rotated_img,new_rect.topleft)

    def GetMask(self):
        return pygame.mask.from_surface(self.img)

class Pipe:
    def __init__(self,x,vel):
        self.x=x
        self.height=0
        self.gap=WIN_HEIGHT/4

        self.top=None
        self.bottom=None
        self.Pipe_Top=pygame.transform.flip(PIPE_IMG,False,True)
        self.Pipe_Bottom=PIPE_IMG
        self.VEL=vel

        self.passed=False
        self.Set_Height()

    def Set_Height(self):
        self.height=random.randrange(round(WIN_HEIGHT/10),round(WIN_HEIGHT*3/5))
        self.top=self.height-self.Pipe_Top.get_height()
        self.bottom=self.height+self.gap


    def Move(self):
        self.x-=self.VEL

    def Draw(self,surface):
        surface.blit(self.Pipe_Top,(self.x,self.top))
        surface.blit(self.Pipe_Bottom,(self.x,self.bottom))
    
    def Collision(self,bird):
        bird_mask=bird.GetMask()
        top_mask=pygame.mask.from_surface(self.Pipe_Top)
        bottom_mask=pygame.mask.from_surface(self.Pipe_Bottom)

        top_offset=(self.x-bird.x, round(self.top-bird.y))
        bottom_offset=(self.x-bird.x, round(self.bottom-bird.y))

        b_point=bird_mask.overlap(bottom_mask,bottom_offset)
        t_point=bird_mask.overlap(top_mask,top_offset)

        if t_point or b_point: return True

        return False

class Base:
    WIDTH=BASE_IMG.get_width()
    HEIGHT=BASE_IMG.get_height()
    IMG=BASE_IMG
    
    def __init__(self,vel):
        self.x1=0
        self.x2=self.WIDTH
        self.VEL=vel
    
    def Move(self):
        self.x1-=self.VEL
        self.x2-=self.VEL

        if self.x1+self.WIDTH<0:
            self.x1=self.x2+self.WIDTH
        elif self.x2+self.WIDTH<0:
            self.x2=self.x1+self.WIDTH
        
    def Draw(self,surface):
        surface.blit(self.IMG,(self.x1,WIN_HEIGHT-self.HEIGHT))
        surface.blit(self.IMG,(self.x2,WIN_HEIGHT-self.HEIGHT))
        
def Draw_Window(surface,birds,pipes,base,score,gen):
    surface.blit(BG_IMG,(0,0))

    for pipe in pipes:
        pipe.Draw(surface)
    
    text=STAT_FONT.render("Score: "+ str(score), 1, (255,255,255))
    surface.blit(text, (WIN_WIDTH-5-text.get_width(),WIN_HEIGHT/30))

    text=STAT_FONT.render("Gen: "+ str(gen), 1, (255,255,255))
    surface.blit(text, (WIN_WIDTH/20,WIN_HEIGHT/30))


    base.Draw(surface)
    text=STAT_FONT.render("Birds Alive: "+ str(len(birds)), 1, (0,0,0))
    surface.blit(text, (WIN_WIDTH/20,0.9*WIN_HEIGHT))

    for bird in birds:
        bird[2].Draw(surface)
    pygame.display.update()

def Move(birds,pipes,base):
    for bird in birds:
        bird[2].Move()
    for pipe in pipes:
        pipe.Move()
    base.Move()

def Main(genomes,config):
    global GEN
    GEN+=1
    #time.sleep(5)
    birds=[]#[ [network,genome,bird obj] ]

    for _,g in genomes:
        g.fitness=0
        birds.append([neat.nn.FeedForwardNetwork.create(g,config),g,Bird(WIN_WIDTH/3,WIN_HEIGHT/3)])
        
    VEL=10
    base=Base(VEL)
    pipes=[Pipe(WIN_WIDTH+150,VEL)]
    win=pygame.display.set_mode((WIN_WIDTH,WIN_HEIGHT))
    clock=pygame.time.Clock()

    score=0

    while True:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit()
                quit()

        ExtraPipes=[]
        add_pipe=False
        for pipe in pipes:
            for bird in birds:

                if pipe.Collision(bird[2]):
                    bird[1].fitness-=1
                    birds.remove(bird)
                if not pipe.passed and pipe.x +pipe.Pipe_Top.get_width()<bird[2].x:
                    pipe.passed=True
                    add_pipe=True

            if pipe.x+pipe.Pipe_Top.get_width()<0:
                ExtraPipes.append(pipe)
            
        Pipe_INDX=0
        if len(birds)>0 and len(pipes)>1:
            ''' if pipes[0].passed:#atmost 2 pipes at a time
                Pipe_INDX=1'''
            for x,pipe in enumerate(pipes): #check if this works fine for general case
                if not pipe.passed:
                    Pipe_INDX=x
                    break
        
      

        for bird in birds:
            bird[1].fitness+=0.1
            output=bird[0].activate((bird[2].y, abs(bird[2].y - pipes[Pipe_INDX].height),abs(bird[2].y - pipes[Pipe_INDX].bottom)))
            if output[0]>0.3:
                bird[2].Jump()
        
        Move(birds,pipes,base)

        if add_pipe:
            VEL+=3/VEL
            #for bird in birds: bird[2].ANIMATION_TIME-=bird[2].ANIMATION_TIME/1000 #visual effect
            score+=1
            pipes.append(Pipe(WIN_WIDTH+100,VEL))
            for bird in birds:
                bird[1].fitness+=5
        for r in ExtraPipes:
            pipes.remove(r)

        for bird in birds:
            if bird[2].y+bird[2].img.get_height()>=0.8*WIN_HEIGHT or bird[2].y<0:
                bird[1].fitness-=1
                birds.remove(bird)
        
        if len(birds)==0:
            break
        if len(birds)==1 and score>50:
            break
        Draw_Window(win,birds,pipes,base,score,GEN)
    
def run(config_path):
    config=neat.config.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    Pop=neat.Population(config)
    Pop.add_reporter((neat.StdOutReporter(True)))
    Pop.add_reporter(neat.StatisticsReporter())

    winner=Pop.run(Main,50)
    
    Trained_Network=open("Trained Network","wb")
    pickle.dump(winner,Trained_Network)
    Trained_Network.close()

    print('\nBest genome:\n{!s}'.format(winner))
  
if __name__=="__main__":
    local_dir=os.path.dirname(__file__)
    config_path=os.path.join(local_dir,"config-feedforward.txt")

    run(config_path)



print("\n\n*******Running the Best Trained Genome*******\n\n")

time.sleep(3)
f=open("Trained Network",'rb')
Trained_Network=pickle.load(f)

local_dir=os.path.dirname(__file__)
config_path=os.path.join(local_dir,"config-feedforward.txt")
config=neat.config.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

def main(genome,config):
    bird=[neat.nn.FeedForwardNetwork.create(genome,config),genome,Bird(WIN_WIDTH/3,WIN_HEIGHT/3)]
        
    VEL=10
    base=Base(VEL)
    pipes=[Pipe(WIN_WIDTH+150,VEL)]
    win=pygame.display.set_mode((WIN_WIDTH,WIN_HEIGHT))
    clock=pygame.time.Clock()

    score=0
    run=True

    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                run=False

        ExtraPipes=[]
        add_pipe=False
        for pipe in pipes:
                if pipe.Collision(bird[2]):
                    run=False
                if not pipe.passed and pipe.x +pipe.Pipe_Top.get_width()<bird[2].x:
                    pipe.passed=True
                    add_pipe=True

                if pipe.x+pipe.Pipe_Top.get_width()<0:
                    ExtraPipes.append(pipe)
            
        Pipe_INDX=0
        if len(pipes)>1:
            for x,pipe in enumerate(pipes): #check if this works fine for general case
                if not pipe.passed:
                    Pipe_INDX=x
                    break
        
      

      
        output=bird[0].activate((bird[2].y, abs(bird[2].y - pipes[Pipe_INDX].height),abs(bird[2].y - pipes[Pipe_INDX].bottom)))
        if output[0]>0.3:
            bird[2].Jump()
        
        Move([bird],pipes,base)

        if add_pipe:
            VEL+=3/VEL
            bird[2].ANIMATION_TIME-=bird[2].ANIMATION_TIME/1000 #visual effect
            score+=1
            pipes.append(Pipe(WIN_WIDTH+100,VEL))

        for r in ExtraPipes:
            pipes.remove(r)

        
        if bird[2].y+bird[2].img.get_height()>=0.8*WIN_HEIGHT or bird[2].y<0:
            run=False
        
        
        Draw_Window(win,[bird],pipes,base,score,GEN)

main(Trained_Network,config)
pygame.quit()
quit()